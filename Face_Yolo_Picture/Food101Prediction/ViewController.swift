//
//  ViewController.swift
//  Food101Prediction
//
//  Created by Philipp Gabriel on 21.06.17.
//  Copyright © 2017 Philipp Gabriel. All rights reserved.
//

import UIKit
import CoreML

class ViewController: UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate {
    

    @IBOutlet weak var imageView: ScaledHeightImageView!
    @IBOutlet weak var percentage: UILabel!
    
    let yolo = YOLO()
    let drawBoundingBoxes = true
    let ciContext = CIContext()
    var resizedPixelBuffers: [CVPixelBuffer?] = []
    var resizedPixelBuffersGender: [CVPixelBuffer?] = []
    var resizedPixelBuffersAge: [CVPixelBuffer?] = []

    public static let maxBoundingBoxes = 10
    let confidenceThreshold: Float = 0.3
    let iouThreshold: Float = 0.5
    
    var boundingBoxes = [BoundingBox]()
    var colors: [UIColor] = []

    override func viewDidLoad() {
        super.viewDidLoad()
        setUpBoundingBoxes()
        setUpCoreImage()
        for box in self.boundingBoxes {
            box.addToLayer(self.imageView.layer)
        }
    }
    func setUpCoreImage() {
        // Since we might be running several requests in parallel, we also need
        // to do the resizing in different pixel buffers or we might overwrite a
        // pixel buffer that's already in use.
        for _ in 0..<YOLO.maxBoundingBoxes*2 {
            var resizedPixelBuffer: CVPixelBuffer?
            let status = CVPixelBufferCreate(nil, YOLO.inputWidth, YOLO.inputHeight,
                                             kCVPixelFormatType_32BGRA, nil,
                                             &resizedPixelBuffer)
            
            if status != kCVReturnSuccess {
                print("Error: could not create resized pixel buffer", status)
            }
            resizedPixelBuffers.append(resizedPixelBuffer)
        }
        for _ in 0..<2 {
            var resizedPixelBufferGender: CVPixelBuffer?
            let status = CVPixelBufferCreate(nil, YOLO.inputWidthGender, YOLO.inputHeightGender,
                                             kCVPixelFormatType_32BGRA, nil,
                                             &resizedPixelBufferGender)
            
            if status != kCVReturnSuccess {
                print("Error: could not create resized pixel buffer", status)
            }
            resizedPixelBuffersGender.append(resizedPixelBufferGender)
        }
        for _ in 0..<2 {
            var resizedPixelBufferGender: CVPixelBuffer?
            let status = CVPixelBufferCreate(nil, YOLO.inputWidthGender, YOLO.inputHeightGender,
                                             kCVPixelFormatType_32BGRA, nil,
                                             &resizedPixelBufferGender)
            
            if status != kCVReturnSuccess {
                print("Error: could not create resized pixel buffer", status)
            }
            resizedPixelBuffersAge.append(resizedPixelBufferGender)
        }
    }
    func setUpBoundingBoxes() {
        for _ in 0..<YOLO.maxBoundingBoxes*2 {
            boundingBoxes.append(BoundingBox())
        }
        
        // Make colors for the bounding boxes. There is one color for each class,
        // 20 classes in total.
        for r: CGFloat in [0.2, 0.4, 0.6, 0.8, 1.0] {
            for g: CGFloat in [0.3, 0.7] {
                for b: CGFloat in [0.4, 0.8] {
                    let color = UIColor(red: r, green: g, blue: b, alpha: 1)
                    colors.append(color)
                }
            }
        }
    }
    
    @IBAction func buttonPressed(_ sender: Any) {
        let alert = UIAlertController(title: nil, message: nil, preferredStyle: .actionSheet)
        let imagePickerView = UIImagePickerController()
        imagePickerView.delegate = self

        alert.addAction(UIAlertAction(title: "Choose Image", style: .default) { _ in
            imagePickerView.sourceType = .photoLibrary
            self.present(imagePickerView, animated: true, completion: nil)
        })

        alert.addAction(UIAlertAction(title: "Take Image", style: .default) { _ in
            imagePickerView.sourceType = .camera
            self.present(imagePickerView, animated: true, completion: nil)
        })

        alert.addAction(UIAlertAction(title: "Cancel", style: .cancel, handler: nil))
        self.present(alert, animated: true, completion: nil)
    }

    @objc internal func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
        dismiss(animated: true, completion: nil)
    }

    @objc internal func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [String: Any]) {
        dismiss(animated: true, completion: nil)

        guard let image = info["UIImagePickerControllerOriginalImage"] as? UIImage else {
            return
        }

        processImage(image)
    }

    
    func processImage(_ image: UIImage) {
        imageView.image = image
        predict(image:image)
    }
    func predict(image: UIImage) {
        if let pixelBuffer = image.pixelBuffer(width: Int(image.size.width), height: Int(image.size.height)) {
            print("pixelBuffer", CVPixelBufferGetWidth(pixelBuffer),CVPixelBufferGetHeight(pixelBuffer))
            predict(pixelBuffer: pixelBuffer, inflightIndex: 0)
        }
    }
    
    func predict(pixelBuffer: CVPixelBuffer, inflightIndex: Int) {
        // Measure how long it takes to predict a single video frame.
        // Resize the input with Core Image to 288x288.
        if let resizedPixelBuffer = resizedPixelBuffers[inflightIndex] {
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            print("ciImage size", ciImage)
            let sx = CGFloat(YOLO.inputWidth) / CGFloat(CVPixelBufferGetWidth(pixelBuffer))
            let sy = CGFloat(YOLO.inputHeight) / CGFloat(CVPixelBufferGetHeight(pixelBuffer))
            print("sx, sy",sx,sy)
            let scaleTransform = CGAffineTransform(scaleX: sx, y: sy)
            let scaledImage = ciImage.transformed(by: scaleTransform)
            ciContext.render(scaledImage, to: resizedPixelBuffer)
            // Give the resized input to our model.
            //        print("the prediction result:",  try? yolo.predict(image: resizedPixelBuffer))
            if let result = try? yolo.predict(image: resizedPixelBuffer),
                let boundingBoxes = result {
                let Finallabels = predictOnCroppedImage(boundingBoxes,resizedPixelBuffer)
                print("The Finallabels are:",Finallabels)
                showOnMainThread(boundingBoxes, pixelBuffer, Finallabels)
            } else {
                print("BOGUS")
            }
            //        print("the boundingBoxes", boundingBoxes)
            print("enter the face predict funtion and the boundingBoxes number", boundingBoxes.count)
        }
        
    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    func predictOnCroppedImage(_ boundingBoxes: [YOLO.Prediction], _ pixelBuffer: CVPixelBuffer) -> [[String]]{
        var finalList = [[String]]()
        var labelsResult = [String]()
        var labelsAgeResult = [String]()
        for i in 0..<boundingBoxes.count {
            if i < boundingBoxes.count {
                let prediction = boundingBoxes[i]
                
                // The predicted bounding box is in the coordinate space of the input
                // image, which is a square image of 288x288 pixels. We want to show it
                // on the video preview, which is as wide as the screen and has a 16:9
                // aspect ratio. The video preview also may be letterboxed at the top
                // and bottom.
                //                print("pixelBuffer size to crop:", CVPixelBufferGetWidth(pixelBuffer),CVPixelBufferGetHeight(pixelBuffer))
                let width = CGFloat(CVPixelBufferGetWidth(pixelBuffer))
                let height = CGFloat(CVPixelBufferGetHeight(pixelBuffer))
                //                print("View size:",width,height)
                let scaleX = width / CGFloat(YOLO.inputWidth)
                let scaleY = height / CGFloat(YOLO.inputHeight)
                let top = (view.bounds.height - height) / 2
                
                // Translate and scale the rectangle to our own coordinate system.
                var rect = prediction.rect
                print("the original rect",rect)
//                rect.origin.x *= scaleX
//                rect.origin.y *= scaleY
//                rect.origin.y += top
//                rect.size.width *= scaleX
//                rect.size.height *= scaleY
                print("rect:",rect.origin.x,rect.origin.y,rect.size.width,rect.size.height)
                print("scale:",scaleX,scaleY)
                
                if (Int(rect.size.width*1.2) > 1) && (Int(rect.size.height*1.2) > 1) && (Int(rect.origin.x) >= 0) && (Int(rect.origin.y) >= 0){
                    if let predictImage = try? resizePixelBufferWithCrop(pixelBuffer, cropX: Int(rect.origin.x),
                                                                         cropY: Int(rect.origin.y),
                                                                         cropWidth: Int(rect.size.width*1.2),
                                                                         cropHeight: Int(rect.size.height*1.2),
                                                                         scaleWidth: Int(1),
                                                                         scaleHeight: Int(1)) {
                        let labelResult = gendersPredict(pixelBuffer: predictImage!)
                        labelsResult.append(labelResult )
                        let ageResult = agesPredict(pixelBuffer: predictImage!)
                        labelsAgeResult.append(ageResult)
                        
                        //                        print("labelResult:",labelResult ?? " ")
                    } else {
                        print("Failed crop the image.")
                        let labelResult = " "
                        labelsResult.append(labelResult )
                        let ageResult = " "
                        labelsAgeResult.append(ageResult)
                    }
                } else {
                    let labelResult = " "
                    labelsResult.append(labelResult )
                    let ageResult = " "
                    labelsAgeResult.append(ageResult)
                }
            }
        }
        finalList.append(labelsResult)
        finalList.append(labelsAgeResult)
        return finalList
    }
    
    func gendersPredict(pixelBuffer: CVPixelBuffer) -> String {
        // Measure how long it takes to predict a single video frame.
        //        let startTime = CACurrentMediaTime()
        // This is an alternative way to resize the image (using vImage):
        //if let resizedPixelBuffer = resizePixelBuffer(pixelBuffer,
        //                                              width: YOLO.inputWidth,
        //                                              height: YOLO.inputHeight) {
        //    self.genderLabels.removeAll()
        // Resize the input with Core Image to 288x288.
        if let resizedPixelBufferGender = resizedPixelBuffersGender[0] {
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let sx = CGFloat(YOLO.inputWidthGender) / CGFloat(CVPixelBufferGetWidth(pixelBuffer))
            let sy = CGFloat(YOLO.inputHeightGender) / CGFloat(CVPixelBufferGetHeight(pixelBuffer))
            let scaleTransform = CGAffineTransform(scaleX: sx, y: sy)
            let scaledImage = ciImage.transformed(by: scaleTransform)
            ciContext.render(scaledImage, to: resizedPixelBufferGender)
            // Give the resized input to our model.
            //        print("the prediction result:",  try? yolo.predict(image: resizedPixelBuffer))
            if let result = try? yolo.genderPredict(image: resizedPixelBufferGender){
                print("the final gender is:", result ?? " ")
                return result!
            } else {
                print("BOGUS")
                return "didnotgettheresult"
            }
            //            print("enter the face predict funtion and the boundingBoxes number", boundingBoxes.count)
        }
        return "didnotgettheresult"
    }
    
    func agesPredict(pixelBuffer: CVPixelBuffer) -> String {
        // Measure how long it takes to predict a single video frame.
        //        let startTime = CACurrentMediaTime()
        // This is an alternative way to resize the image (using vImage):
        //if let resizedPixelBuffer = resizePixelBuffer(pixelBuffer,
        //                                              width: YOLO.inputWidth,
        //                                              height: YOLO.inputHeight) {
        //    self.genderLabels.removeAll()
        // Resize the input with Core Image to 288x288.
        if let resizedPixelBufferAge = resizedPixelBuffersAge[0] {
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let sx = CGFloat(YOLO.inputWidthGender) / CGFloat(CVPixelBufferGetWidth(pixelBuffer))
            let sy = CGFloat(YOLO.inputHeightGender) / CGFloat(CVPixelBufferGetHeight(pixelBuffer))
            let scaleTransform = CGAffineTransform(scaleX: sx, y: sy)
            let scaledImage = ciImage.transformed(by: scaleTransform)
            ciContext.render(scaledImage, to: resizedPixelBufferAge)
            // Give the resized input to our model.
            //        print("the prediction result:",  try? yolo.predict(image: resizedPixelBuffer))
            if let result = try? yolo.agePredict(image: resizedPixelBufferAge){
                print("the final age is:", result ?? " ")
                return result!
            } else {
                print("BOGUS")
                return "didnotgettheresult"
            }
            //            print("enter the face predict funtion and the boundingBoxes number", boundingBoxes.count)
        }
        return "didnotgettheresult"
    }
    
    func showOnMainThread(_ boundingBoxes: [YOLO.Prediction],_ pixelBuffer: CVPixelBuffer, _ Finallabels: [[String]]) {
        if drawBoundingBoxes {
            DispatchQueue.main.async {
                // For debugging, to make sure the resized CVPixelBuffer is correct.
                //var debugImage: CGImage?
                //VTCreateCGImageFromCVPixelBuffer(resizedPixelBuffer, nil, &debugImage)
                //self.debugImageView.image = UIImage(cgImage: debugImage!)
                
                self.show(predictions: boundingBoxes,pixelBuffer,Finallabels: Finallabels)
                
                
                
            }
        }
    }
    
    func show(predictions: [YOLO.Prediction], _ pixelBuffer: CVPixelBuffer, Finallabels: [[String]]) {
        for i in 0..<boundingBoxes.count {
            if i < predictions.count {
                let prediction = predictions[i]
                
                // The predicted bounding box is in the coordinate space of the input
                // image, which is a square image of 416x416 pixels. We want to show it
                // on the video preview, which is as wide as the screen and has a 16:9
                // aspect ratio. The video preview also may be letterboxed at the top
                // and bottom.
                let width = imageView.bounds.width
                let height = imageView.bounds.height
                let scaleX = width / CGFloat(YOLO.inputWidth)
                let scaleY = height / CGFloat(YOLO.inputHeight)
                let top = (imageView.bounds.height - height) / 2
//                print("imageview size",width,height)
//                print("image size",imageView.image?.size)
//                print("scale",scaleX,scaleY)
                // Translate and scale the rectangle to our own coordinate system.
                var rect = prediction.rect
                print("the original rect",rect)
                rect.origin.x *= scaleX
                rect.origin.y *= scaleY
                rect.origin.y += top
                rect.size.width *= scaleX
                rect.size.height *= scaleY
                //        print("width,height",rect.size.width, rect.size.height)
                //        print("show the box rect:",rect)
                // Show the bounding box. , labels[0]
                //        let genderLabel = self.genderLabels[i]
                //        let label = String(format: "%@ %.1f", genderLabel, prediction.score * 100)
                let color = colors[3]
                //        print("show the box label and color:",label,color)
                boundingBoxes[i].show(frame: rect, label: "\(Finallabels[0][i])", color: color)
                print("enter the show function and showing the box",i)
            } else {
                boundingBoxes[i].hide()
            }
        }
    }
   
}

class ScaledHeightImageView: UIImageView {
    
    override var intrinsicContentSize: CGSize {
        
        if let myImage = self.image {
            let myImageWidth = myImage.size.width
            let myImageHeight = myImage.size.height
            let myViewWidth = self.frame.size.width
            
            let ratio = myViewWidth/myImageWidth
            let scaledHeight = myImageHeight * ratio
            
            return CGSize(width: myImageWidth, height: myImageHeight)
        }
        
        return CGSize(width: -1.0, height: -1.0)
    }
    
}
