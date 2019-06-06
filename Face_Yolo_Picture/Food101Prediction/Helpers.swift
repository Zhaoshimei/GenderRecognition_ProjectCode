import Foundation
import UIKit
import CoreML
import Accelerate
import CoreImage

// The labels for the 20 classes.
//let labels = [
//    "face"
//]

let anchors: [Float] = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

/**
 Removes bounding boxes that overlap too much with other boxes that have
 a higher score.
 
 Based on code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/non_max_suppression_op.cc
 
 - Parameters:
 - boxes: an array of bounding boxes and their scores
 - limit: the maximum number of boxes that will be selected
 - threshold: used to decide whether boxes overlap too much
 */
func nonMaxSuppression(boxes: [YOLO.Prediction], limit: Int, threshold: Float) -> [YOLO.Prediction] {
    
    // Do an argsort on the confidence scores, from high to low.
    let sortedIndices = boxes.indices.sorted { boxes[$0].score > boxes[$1].score }
    
    var selected: [YOLO.Prediction] = []
    var active = [Bool](repeating: true, count: boxes.count)
    var numActive = active.count
    
    // The algorithm is simple: Start with the box that has the highest score.
    // Remove any remaining boxes that overlap it more than the given threshold
    // amount. If there are any boxes left (i.e. these did not overlap with any
    // previous boxes), then repeat this procedure, until no more boxes remain
    // or the limit has been reached.
    outer: for i in 0..<boxes.count {
        if active[i] {
            let boxA = boxes[sortedIndices[i]]
            selected.append(boxA)
            if selected.count >= limit { break }
            
            for j in i+1..<boxes.count {
                if active[j] {
                    let boxB = boxes[sortedIndices[j]]
                    if IOU(a: boxA.rect, b: boxB.rect) > threshold {
                        active[j] = false
                        numActive -= 1
                        if numActive <= 0 { break outer }
                    }
                }
            }
        }
    }
    return selected
}

/**
 Computes intersection-over-union overlap between two bounding boxes.
 */
public func IOU(a: CGRect, b: CGRect) -> Float {
    let areaA = a.width * a.height
    if areaA <= 0 { return 0 }
    
    let areaB = b.width * b.height
    if areaB <= 0 { return 0 }
    
    let intersectionMinX = max(a.minX, b.minX)
    let intersectionMinY = max(a.minY, b.minY)
    let intersectionMaxX = min(a.maxX, b.maxX)
    let intersectionMaxY = min(a.maxY, b.maxY)
    let intersectionArea = max(intersectionMaxY - intersectionMinY, 0) *
        max(intersectionMaxX - intersectionMinX, 0)
    return Float(intersectionArea / (areaA + areaB - intersectionArea))
}

extension Array where Element: Comparable {
    /**
     Returns the index and value of the largest element in the array.
     */
    public func argmax() -> (Int, Element) {
        precondition(self.count > 0)
        var maxIndex = 0
        var maxValue = self[0]
        for i in 1..<self.count {
            if self[i] > maxValue {
                maxValue = self[i]
                maxIndex = i
            }
        }
        return (maxIndex, maxValue)
    }
}

/**
 Logistic sigmoid.
 */
public func sigmoid(_ x: Float) -> Float {
    return 1 / (1 + exp(-x))
}

/**
 Computes the "softmax" function over an array.
 
 Based on code from https://github.com/nikolaypavlov/MLPNeuralNet/
 
 This is what softmax looks like in "pseudocode" (actually using Python
 and numpy):
 
 x -= np.max(x)
 exp_scores = np.exp(x)
 softmax = exp_scores / np.sum(exp_scores)
 
 First we shift the values of x so that the highest value in the array is 0.
 This ensures numerical stability with the exponents, so they don't blow up.
 */
public func softmax(_ x: [Float]) -> [Float] {
    var x = x
    let len = vDSP_Length(x.count)
    
    // Find the maximum value in the input array.
    var max: Float = 0
    vDSP_maxv(x, 1, &max, len)
    
    // Subtract the maximum from all the elements in the array.
    // Now the highest value in the array is 0.
    max = -max
    vDSP_vsadd(x, 1, &max, &x, 1, len)
    
    // Exponentiate all the elements in the array.
    var count = Int32(x.count)
    vvexpf(&x, x, &count)
    
    // Compute the sum of all exponentiated values.
    var sum: Float = 0
    vDSP_sve(x, 1, &sum, len)
    
    // Divide each element by the sum. This normalizes the array contents
    // so that they all add up to 1.
    vDSP_vsdiv(x, 1, &sum, &x, 1, len)
    
    return x
}

/**
 Creates a RGB pixel buffer of the specified width and height.
 */
public func createPixelBuffer(width: Int, height: Int) -> CVPixelBuffer? {
    var pixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreate(nil, width, height,
                                     kCVPixelFormatType_32BGRA, nil,
                                     &pixelBuffer)
    if status != kCVReturnSuccess {
        print("Error: could not create pixel buffer", status)
        return nil
    }
    return pixelBuffer
}

/**
 First crops the pixel buffer, then resizes it.
 */
public func resizePixelBufferWithCrop(_ srcPixelBuffer: CVPixelBuffer,
                                      cropX: Int,
                                      cropY: Int,
                                      cropWidth: Int,
                                      cropHeight: Int,
                                      scaleWidth: Int,
                                      scaleHeight: Int) -> CVPixelBuffer? {
    print("enter the crop function.")
    print("crop",cropX,cropY,cropWidth,cropHeight)
    let flags = CVPixelBufferLockFlags(rawValue: 0)
    guard kCVReturnSuccess == CVPixelBufferLockBaseAddress(srcPixelBuffer, flags) else {
        return nil
    }
    defer { CVPixelBufferUnlockBaseAddress(srcPixelBuffer, flags) }
    
    guard let srcData = CVPixelBufferGetBaseAddress(srcPixelBuffer) else {
        print("Error: could not get pixel buffer base address")
        return nil
    }
    let srcBytesPerRow = CVPixelBufferGetBytesPerRow(srcPixelBuffer)
    let offset = cropY*srcBytesPerRow + cropX*4
    var srcBuffer = vImage_Buffer(data: srcData.advanced(by: offset),
                                  height: vImagePixelCount(cropHeight),
                                  width: vImagePixelCount(cropWidth),
                                  rowBytes: srcBytesPerRow)
    
    let destBytesPerRow = scaleWidth*4
    guard let destData = malloc(scaleHeight*destBytesPerRow) else {
        print("Error: out of memory")
        return nil
    }
    var destBuffer = vImage_Buffer(data: destData,
                                   height: vImagePixelCount(scaleHeight),
                                   width: vImagePixelCount(scaleWidth),
                                   rowBytes: destBytesPerRow)
    
    let error = vImageScale_ARGB8888(&srcBuffer, &destBuffer, nil, vImage_Flags(0))
    if error != kvImageNoError {
        print("Error:", error)
        print("this is where the error comes from.")
        free(destData)
        return nil
    }
    
    let releaseCallback: CVPixelBufferReleaseBytesCallback = { _, ptr in
        if let ptr = ptr {
            free(UnsafeMutableRawPointer(mutating: ptr))
        }
    }
    
    let pixelFormat = CVPixelBufferGetPixelFormatType(srcPixelBuffer)
    var dstPixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreateWithBytes(nil, scaleWidth, scaleHeight,
                                              pixelFormat, destData,
                                              destBytesPerRow, releaseCallback,
                                              nil, nil, &dstPixelBuffer)
    if status != kCVReturnSuccess {
        print("Error: could not create new pixel buffer")
        free(destData)
        return nil
    }
    print("crop the image to dstPixelBuffer:", dstPixelBuffer as Any)
    return dstPixelBuffer
}

/**
 Resizes a CVPixelBuffer to a new width and height.
 */
public func resizePixelBufferOne(_ pixelBuffer: CVPixelBuffer, width: Int, height: Int) -> CVPixelBuffer? {
    return resizePixelBufferWithCrop(pixelBuffer, cropX: 0, cropY: 0,
                                     cropWidth: CVPixelBufferGetWidth(pixelBuffer),
                                     cropHeight: CVPixelBufferGetHeight(pixelBuffer),
                                     scaleWidth: width, scaleHeight: height)
}

/**
 Resizes a CVPixelBuffer to a new width and height.
 */
public func resizePixelBufferAnother(_ pixelBuffer: CVPixelBuffer,
                                     width: Int, height: Int,
                                     output: CVPixelBuffer, context: CIContext) {
    let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
    let sx = CGFloat(width) / CGFloat(CVPixelBufferGetWidth(pixelBuffer))
    let sy = CGFloat(height) / CGFloat(CVPixelBufferGetHeight(pixelBuffer))
    let scaleTransform = CGAffineTransform(scaleX: sx, y: sy)
    let scaledImage = ciImage.transformed(by: scaleTransform)
    context.render(scaledImage, to: output)
}

/**
 Rotates CVPixelBuffer by the provided factor of 90 counterclock-wise.
 */
public func rotate90PixelBuffer(_ srcPixelBuffer: CVPixelBuffer, factor: UInt8) -> CVPixelBuffer? {
    let flags = CVPixelBufferLockFlags(rawValue: 0)
    guard kCVReturnSuccess == CVPixelBufferLockBaseAddress(srcPixelBuffer, flags) else {
        return nil
    }
    defer { CVPixelBufferUnlockBaseAddress(srcPixelBuffer, flags) }
    
    guard let srcData = CVPixelBufferGetBaseAddress(srcPixelBuffer) else {
        print("Error: could not get pixel buffer base address")
        return nil
    }
    let sourceWidth = CVPixelBufferGetWidth(srcPixelBuffer)
    let sourceHeight = CVPixelBufferGetHeight(srcPixelBuffer)
    var destWidth = sourceHeight
    var destHeight = sourceWidth
    var color = UInt8(0)
    
    if factor % 2 == 0 {
        destWidth = sourceWidth
        destHeight = sourceHeight
    }
    
    let srcBytesPerRow = CVPixelBufferGetBytesPerRow(srcPixelBuffer)
    var srcBuffer = vImage_Buffer(data: srcData,
                                  height: vImagePixelCount(sourceHeight),
                                  width: vImagePixelCount(sourceWidth),
                                  rowBytes: srcBytesPerRow)
    
    let destBytesPerRow = destWidth*4
    guard let destData = malloc(destHeight*destBytesPerRow) else {
        print("Error: out of memory")
        return nil
    }
    var destBuffer = vImage_Buffer(data: destData,
                                   height: vImagePixelCount(destHeight),
                                   width: vImagePixelCount(destWidth),
                                   rowBytes: destBytesPerRow)
    
    let error = vImageRotate90_ARGB8888(&srcBuffer, &destBuffer, factor, &color, vImage_Flags(0))
    if error != kvImageNoError {
        print("Error:", error)
        free(destData)
        return nil
    }
    
    let releaseCallback: CVPixelBufferReleaseBytesCallback = { _, ptr in
        if let ptr = ptr {
            free(UnsafeMutableRawPointer(mutating: ptr))
        }
    }
    
    let pixelFormat = CVPixelBufferGetPixelFormatType(srcPixelBuffer)
    var dstPixelBuffer: CVPixelBuffer?
    let status = CVPixelBufferCreateWithBytes(nil, destWidth, destHeight,
                                              pixelFormat, destData,
                                              destBytesPerRow, releaseCallback,
                                              nil, nil, &dstPixelBuffer)
    if status != kCVReturnSuccess {
        print("Error: could not create new pixel buffer")
        free(destData)
        return nil
    }
    return dstPixelBuffer
}
