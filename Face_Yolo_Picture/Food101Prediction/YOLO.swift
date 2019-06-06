import Foundation
import UIKit
import CoreML

class YOLO {
  public static let inputWidth = 288
  public static let inputHeight = 288
  public static let inputWidthGender = 227
  public static let inputHeightGender = 227
  public static let maxBoundingBoxes = 10

  // Tweak these values to get more or fewer predictions.
  let confidenceThreshold: Float = 0.3
  let iouThreshold: Float = 0.5

  struct Prediction {
//    let classIndex: Int
    let score: Float
    let rect: CGRect
  }

  let model = face_detection()

  public init() { }
    
// detect face
  public func predict(image: CVPixelBuffer) throws -> [Prediction]? {
    if let output = try? model.prediction(image: image) {
      return computeBoundingBoxes(features: output.output1)
//+computeBoundingBoxesSmallGrid(features: output.output2)
    } else {
      return nil
    }
  }
//by Shimei Zhao begin, predict face
    let genderModel = GenderNet()
    public func genderPredict(image: CVPixelBuffer) throws -> String? {
        print("image is nil",image != nil)
        if (try? genderModel.prediction(data: image)) != nil {
            let result = try? genderModel.prediction(data: image)
            let genderLabel: String = (result?.classLabel)!
            print("gender prediction result",genderLabel)
            return genderLabel
        } else {
            return nil
        }
    }
    let ageModel = AgeNet()
    public func agePredict(image: CVPixelBuffer) throws -> String? {
        print("image is nil",image != nil)
        if (try? ageModel.prediction(data: image)) != nil {
            let result = try? ageModel.prediction(data: image)
            let ageLabel: String = (result?.classLabel)!
            print("age prediction result",ageLabel)
            return ageLabel
        } else {
            return nil
        }
    }
//by Shimei Zhao end


  public func computeBoundingBoxes(features: MLMultiArray) -> [Prediction] {
    assert(features.count == 5*9*9)

    var predictions = [Prediction]()

    let blockSize: Float = 32
    let gridHeight = 9
    let gridWidth = 9
//    let boxesPerCell = 0
    let numClasses = 0

    // The 288x288 image is divided into a 9x9 grid. Each of these grid cells
    // will predict 5 bounding boxes (boxesPerCell). A bounding box consists of
    // five data items: x, y, width, height, and a confidence score. Each grid
    // cell also predicts which class each bounding box belongs to.
    //
    // The "features" array therefore contains (numClasses:0 + 5)*boxesPerCell:0
    // values for each grid cell, i.e. 5 channels. The total features array
    // contains 5x9x9 elements.

    // NOTE: It turns out that accessing the elements in the multi-array as
    // `features[[channel, cy, cx] as [NSNumber]].floatValue` is kinda slow.
    // It's much faster to use direct memory access to the features.
    let featurePointer = UnsafeMutablePointer<Double>(OpaquePointer(features.dataPointer))
    let channelStride = features.strides[0].intValue
    let yStride = features.strides[1].intValue
    let xStride = features.strides[2].intValue
    print("channelStride,yStride,xStride:",channelStride,yStride,xStride)
    @inline(__always) func offset(_ channel: Int, _ x: Int, _ y: Int) -> Int {
      return channel*channelStride + y*yStride + x*xStride
    }

    for cy in 0..<gridHeight {
      for cx in 0..<gridWidth {
//        for b in 0..<boxesPerCell {

          // For the first bounding box (b=0) we have to read channels 0-24,
          // for b=1 we have to read channels 25-49, and so on.
          let channel = 0

          // The slow way:
          /*
          let tx = features[[channel    , cy, cx] as [NSNumber]].floatValue
          let ty = features[[channel + 1, cy, cx] as [NSNumber]].floatValue
          let tw = features[[channel + 2, cy, cx] as [NSNumber]].floatValue
          let th = features[[channel + 3, cy, cx] as [NSNumber]].floatValue
          let tc = features[[channel + 4, cy, cx] as [NSNumber]].floatValue
          */

          // The fast way:
          let tx = Float(featurePointer[offset(channel    , cx, cy)])
          let ty = Float(featurePointer[offset(channel + 1, cx, cy)])
          let tw = Float(featurePointer[offset(channel + 2, cx, cy)])
          let th = Float(featurePointer[offset(channel + 3, cx, cy)])
          let tc = Float(featurePointer[offset(channel + 4, cx, cy)])
//        print("tx,ty,tw,th,tc:",tx,ty,tw,th,tc)
          // The predicted tx and ty coordinates are relative to the location
          // of the grid cell; we use the logistic sigmoid to constrain these
          // coordinates to the range 0 - 1. Then we add the cell coordinates
          // (0-12) and multiply by the number of pixels per grid cell (32).
          // Now x and y represent center of the bounding box in the original
          // 288x288 image space.
          let x = (Float(cx) + sigmoid(tx)) * blockSize
          let y = (Float(cy) + sigmoid(ty)) * blockSize

          // The size of the bounding box, tw and th, is predicted relative to
          // the size of an "anchor" box. Here we also transform the width and
          // height into the original 416x416 image space.
          let w = exp(tw) * 288
          let h = exp(th) * 288

          // The confidence value for the bounding box is given by tc. We use
          // the logistic sigmoid to turn this into a percentage.
          let confidence = sigmoid(tc)

          // Gather the predicted classes for this anchor box and softmax them,
          // so we can interpret these numbers as percentages.
//          var classes = [Float](repeating: 0, count: numClasses)
//          for c in 0..<numClasses {
            // The slow way:
            //classes[c] = features[[channel + 5 + c, cy, cx] as [NSNumber]].floatValue

            // The fast way:
//            classes[c] = Float(featurePointer[offset(channel + 5 + c, cx, cy)])
//          }
//          classes = softmax(classes)

          // Find the index of the class with the largest score.
//          let (detectedClass, bestClassScore) = classes.argmax()

          // Combine the confidence score for the bounding box, which tells us
          // how likely it is that there is an object in this box (but not what
          // kind of object it is), with the largest class prediction, which
          // tells us what kind of object it detected (but not where).
          let confidenceInClass = confidence
//        print("the confidence for each time:",confidenceInClass)
          // Since we compute 13x13x5 = 845 bounding boxes, we only want to
          // keep the ones whose combined score is over a certain threshold.
          if confidenceInClass > confidenceThreshold {
            let rect = CGRect(x: CGFloat(x - w/2), y: CGFloat(y - h/2),
                              width: CGFloat(w), height: CGFloat(h))

            let prediction = Prediction(score: confidenceInClass,
                                        rect: rect)
            predictions.append(prediction)
          }
//        }
      }
    }

    // We already filtered out any bounding boxes that have very low scores,
    // but there still may be boxes that overlap too much with others. We'll
    // use "non-maximum suppression" to prune those duplicate bounding boxes.
    return nonMaxSuppression(boxes: predictions, limit: YOLO.maxBoundingBoxes, threshold: iouThreshold)
  }
    
    public func computeBoundingBoxesSmallGrid(features: MLMultiArray) -> [Prediction] {
        assert(features.count == 5*18*18)
        
        var predictions = [Prediction]()
        
        let blockSize: Float = 16
        let gridHeight = 18
        let gridWidth = 18
        //    let boxesPerCell = 0
        let numClasses = 0
        
        let featurePointer = UnsafeMutablePointer<Double>(OpaquePointer(features.dataPointer))
        let channelStride = features.strides[0].intValue
        let yStride = features.strides[1].intValue
        let xStride = features.strides[2].intValue
        print("channelStride,yStride,xStride:",channelStride,yStride,xStride)
        @inline(__always) func offset(_ channel: Int, _ x: Int, _ y: Int) -> Int {
            return channel*channelStride + y*yStride + x*xStride
        }
        
        for cy in 0..<gridHeight {
            for cx in 0..<gridWidth {
                //        for b in 0..<boxesPerCell {
                
                // For the first bounding box (b=0) we have to read channels 0-24,
                // for b=1 we have to read channels 25-49, and so on.
                let channel = 0
                
                // The slow way:
                /*
                 let tx = features[[channel    , cy, cx] as [NSNumber]].floatValue
                 let ty = features[[channel + 1, cy, cx] as [NSNumber]].floatValue
                 let tw = features[[channel + 2, cy, cx] as [NSNumber]].floatValue
                 let th = features[[channel + 3, cy, cx] as [NSNumber]].floatValue
                 let tc = features[[channel + 4, cy, cx] as [NSNumber]].floatValue
                 */
                
                // The fast way:
                let tx = Float(featurePointer[offset(channel    , cx, cy)])
                let ty = Float(featurePointer[offset(channel + 1, cx, cy)])
                let tw = Float(featurePointer[offset(channel + 2, cx, cy)])
                let th = Float(featurePointer[offset(channel + 3, cx, cy)])
                let tc = Float(featurePointer[offset(channel + 4, cx, cy)])
                print("tx,ty,tw,th,tc:",tx,ty,tw,th,tc)
                // The predicted tx and ty coordinates are relative to the location
                // of the grid cell; we use the logistic sigmoid to constrain these
                // coordinates to the range 0 - 1. Then we add the cell coordinates
                // (0-12) and multiply by the number of pixels per grid cell (32).
                // Now x and y represent center of the bounding box in the original
                // 416x416 image space.
                let x = (Float(cx) + sigmoid(tx)) * blockSize
                let y = (Float(cy) + sigmoid(ty)) * blockSize
                
                // The size of the bounding box, tw and th, is predicted relative to
                // the size of an "anchor" box. Here we also transform the width and
                // height into the original 416x416 image space.
                let w = exp(tw) * 288
                let h = exp(th) * 288
                
                // The confidence value for the bounding box is given by tc. We use
                // the logistic sigmoid to turn this into a percentage.
                let confidence = sigmoid(tc)
                
                // Gather the predicted classes for this anchor box and softmax them,
                // so we can interpret these numbers as percentages.
                //          var classes = [Float](repeating: 0, count: numClasses)
                //          for c in 0..<numClasses {
                // The slow way:
                //classes[c] = features[[channel + 5 + c, cy, cx] as [NSNumber]].floatValue
                
                // The fast way:
                //            classes[c] = Float(featurePointer[offset(channel + 5 + c, cx, cy)])
                //          }
                //          classes = softmax(classes)
                
                // Find the index of the class with the largest score.
                //          let (detectedClass, bestClassScore) = classes.argmax()
                
                // Combine the confidence score for the bounding box, which tells us
                // how likely it is that there is an object in this box (but not what
                // kind of object it is), with the largest class prediction, which
                // tells us what kind of object it detected (but not where).
                let confidenceInClass = confidence
                print("the confidence for each time:",confidenceInClass)
                // Since we compute 13x13x5 = 845 bounding boxes, we only want to
                // keep the ones whose combined score is over a certain threshold.
                if confidenceInClass > confidenceThreshold {
                    let rect = CGRect(x: CGFloat(x - w/2), y: CGFloat(y - h/2),
                                      width: CGFloat(w), height: CGFloat(h))
                    
                    let prediction = Prediction(score: confidenceInClass,
                                                rect: rect)
                    predictions.append(prediction)
                }
                //        }
            }
        }
        
        // We already filtered out any bounding boxes that have very low scores,
        // but there still may be boxes that overlap too much with others. We'll
        // use "non-maximum suppression" to prune those duplicate bounding boxes.
        return nonMaxSuppression(boxes: predictions, limit: YOLO.maxBoundingBoxes, threshold: iouThreshold)
    }
}
