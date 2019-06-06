import UIKit
import Vision
import AVFoundation
import CoreMedia

class ViewController: UIViewController {
    @IBOutlet weak var videoPreview: UIView!
    @IBOutlet weak var timeLabel: UILabel!
    @IBOutlet weak var debugImageView: UIImageView!
    
    // true: use Vision to drive Core ML, false: use plain Core ML
    let useVision = false
    
    // Disable this to see the energy impact of just running the neural net,
    // otherwise it also counts the GPU activity of drawing the bounding boxes.
    let drawBoundingBoxes = true
    
    // How many predictions we can do concurrently.
    static let maxInflightBuffers = 3
    
    let yolo = YOLO()
    
    var videoCapture: VideoCapture!
    var requests = [VNCoreMLRequest]()
    var startTimes: [CFTimeInterval] = []
    
    var boundingBoxes = [BoundingBox]()
    var colors: [UIColor] = []
    
    let ciContext = CIContext()
    var resizedPixelBuffers: [CVPixelBuffer?] = []
    
    var resizedPixelBuffersGender: [CVPixelBuffer?] = []
    
    var framesDone = 0
    var frameCapturingStartTime = CACurrentMediaTime()
    
    var inflightBuffer = 0
    let semaphore = DispatchSemaphore(value: ViewController.maxInflightBuffers)
    
    // COREML: Gender
    var visionRequestsGender = [VNRequest]()
    //  var genderLabels = [String]()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        self.title = "FaceDetector"
//        let rightSwipe = UISwipeGestureRecognizer(target: self, action: #selector(swipeAction(swipe:)))
//        rightSwipe.direction = UISwipeGestureRecognizer.Direction.right
//        self.view.addGestureRecognizer(rightSwipe)
        timeLabel.text = ""
        
        setUpBoundingBoxes()
        setUpCoreImage()
        setUpVision()
        setUpCamera()
        
        frameCapturingStartTime = CACurrentMediaTime()
        
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        print(#function)
    }
    
    
    // MARK: - Initialization
    
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
    }
    
    func setUpVision() {
        guard let visionModel = try? VNCoreMLModel(for: yolo.model.model) else {
            print("Error: could not create Vision model")
            return
        }
        
        for _ in 0..<ViewController.maxInflightBuffers {
            let request = VNCoreMLRequest(model: visionModel, completionHandler: visionRequestDidComplete)
            
            // NOTE: If you choose another crop/scale option, then you must also
            // change how the BoundingBox objects get scaled when they are drawn.
            // Currently they assume the full input image is used.
            request.imageCropAndScaleOption = .scaleFill
            requests.append(request)
        }
    }
    
    func setUpCamera() {
        videoCapture = VideoCapture()
        videoCapture.delegate = self
        videoCapture.desiredFrameRate = 240
        videoCapture.setUp(sessionPreset: AVCaptureSession.Preset.hd1280x720) { success in
            if success {
                // Add the video preview into the UI.
                if let previewLayer = self.videoCapture.previewLayer {
                    self.videoPreview.layer.addSublayer(previewLayer)
                    self.resizePreviewLayer()
                }
                
                // Add the bounding box layers to the UI, on top of the video preview.
                for box in self.boundingBoxes {
                    box.addToLayer(self.videoPreview.layer)
                }
                
                // Once everything is set up, we can start capturing live video.
                self.videoCapture.start()
            }
        }
    }
    
    // MARK: - UI stuff
    
    override func viewWillLayoutSubviews() {
        super.viewWillLayoutSubviews()
        resizePreviewLayer()
    }
    
    override var preferredStatusBarStyle: UIStatusBarStyle {
        return .lightContent
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        // Pause the view's session
        
    }

    func resizePreviewLayer() {
        videoCapture.previewLayer?.frame = videoPreview.bounds
    }
    
    // MARK: - Doing inference
    
    func predict(image: UIImage) {
        print("nocall")
        print("UIImage:",image)
        if let pixelBuffer = image.pixelBuffer(width: Int(view.bounds.width), height: Int(view.bounds.height)) {
            print("pixelBuffer",CVPixelBufferGetWidth(pixelBuffer),CVPixelBufferGetHeight(pixelBuffer))
            predict(pixelBuffer: pixelBuffer, inflightIndex: 0)
        }
    }
    
    func predict(pixelBuffer: CVPixelBuffer, inflightIndex: Int) {
        // Measure how long it takes to predict a single video frame.
        let startTime = CACurrentMediaTime()
        // Resize the input with Core Image to 288x288.
        if let resizedPixelBuffer = resizedPixelBuffers[inflightIndex] {
            let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
            let sx = CGFloat(YOLO.inputWidth) / CGFloat(CVPixelBufferGetWidth(pixelBuffer))
            let sy = CGFloat(YOLO.inputHeight) / CGFloat(CVPixelBufferGetHeight(pixelBuffer))
            let scaleTransform = CGAffineTransform(scaleX: sx, y: sy)
            let scaledImage = ciImage.transformed(by: scaleTransform)
            ciContext.render(scaledImage, to: resizedPixelBuffer)
            // Give the resized input to our model.
            //        print("the prediction result:",  try? yolo.predict(image: resizedPixelBuffer))
            if let result = try? yolo.predict(image: resizedPixelBuffer),
                let boundingBoxes = result {
                //        print("resizedPixelBuffer:", resizedPixelBuffer)
                //        predictGenderOnBoundingBox(resizedPixelBuffer, boundingBoxes)
                let Glabels = predictOnCroppedImage(boundingBoxes,pixelBuffer)
                print("The Glables are:",Glabels)
                //        let genderP = Glabels
                //        print("The genderP is:", genderP)
                //        let genderP = gendersPredict(pixelBuffer: resizedPixelBuffer)
                let elapsed = CACurrentMediaTime() - startTime
                showOnMainThread(boundingBoxes, elapsed, Glabels)
            } else {
                print("BOGUS")
            }
            //        print("the boundingBoxes", boundingBoxes)
            print("enter the face predict funtion and the boundingBoxes number", boundingBoxes.count)
        }
        
        self.semaphore.signal()
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
        
        self.semaphore.signal()
        return "didnotgettheresult"
    }
    
    
    func predictUsingVision(pixelBuffer: CVPixelBuffer, inflightIndex: Int) {
        // Measure how long it takes to predict a single video frame. Note that
        // predict() can be called on the next frame while the previous one is
        // still being processed. Hence the need to queue up the start times.
        startTimes.append(CACurrentMediaTime())
        
        // Vision will automatically resize the input image.
        let handler = VNImageRequestHandler(cvPixelBuffer: pixelBuffer)
        let request = requests[inflightIndex]
        
        // Because perform() will block until after the request completes, we
        // run it on a concurrent background queue, so that the next frame can
        // be scheduled in parallel with this one.
        DispatchQueue.global().async {
            try? handler.perform([request])
        }
    }
    
    func visionRequestDidComplete(request: VNRequest, error: Error?) {
        if let observations = request.results as? [VNCoreMLFeatureValueObservation],
            let features = observations.first?.featureValue.multiArrayValue {
            let boundingBoxes = yolo.computeBoundingBoxes(features: features)
            let elapsed = CACurrentMediaTime() - startTimes.remove(at: 0)
            
            showOnMainThread(boundingBoxes, elapsed, ["useVisionCompleted"])
        } else {
            print("BOGUS!")
        }
        
        self.semaphore.signal()
    }
    
    func showOnMainThread(_ boundingBoxes: [YOLO.Prediction], _ elapsed: CFTimeInterval,_ Glabel: [String]) {
        if drawBoundingBoxes {
            DispatchQueue.main.async {
                // For debugging, to make sure the resized CVPixelBuffer is correct.
                //var debugImage: CGImage?
                //VTCreateCGImageFromCVPixelBuffer(resizedPixelBuffer, nil, &debugImage)
                //self.debugImageView.image = UIImage(cgImage: debugImage!)
                
                self.show(predictions: boundingBoxes, Glabel: Glabel)
                
                let fps = self.measureFPS()
                self.timeLabel.text = String(format: "Elapsed %.5f seconds - %.2f FPS", elapsed, fps)
            }
        }
    }
    
    func measureFPS() -> Double {
        // Measure how many frames were actually delivered per second.
        framesDone += 1
        let frameCapturingElapsed = CACurrentMediaTime() - frameCapturingStartTime
        let currentFPSDelivered = Double(framesDone) / frameCapturingElapsed
        if frameCapturingElapsed > 1 {
            framesDone = 0
            frameCapturingStartTime = CACurrentMediaTime()
        }
        return currentFPSDelivered
    }
    // used
    func predictOnCroppedImage(_ boundingBoxes: [YOLO.Prediction], _ pixelBuffer: CVPixelBuffer) -> [String]{
        var labelsResult = [String]()
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
                let height = width * 16 / 9
//                print("View size:",width,height)
                let scaleX = width / CGFloat(YOLO.inputWidth)
                let scaleY = height / CGFloat(YOLO.inputHeight)
                let top = (view.bounds.height - height) / 2
                
                // Translate and scale the rectangle to our own coordinate system.
                var rect = prediction.rect
                print("the original rect",rect)
                rect.origin.x *= scaleX
                rect.origin.y *= scaleY
                rect.origin.y += top
                rect.size.width *= scaleX
                rect.size.height *= scaleY
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
                        labelsResult.append(labelResult ?? " ")
//                        print("labelResult:",labelResult ?? " ")
                    } else {
                        print("Failed crop the image.")
                        let labelResult = "Female"
                        labelsResult.append(labelResult ?? "")
                    }
                } else {
                    let labelResult = "Female"
                    labelsResult.append(labelResult ?? "")
                }
            }
        }
        return labelsResult
    }
    
    func show(predictions: [YOLO.Prediction], Glabel: [String]) {
        for i in 0..<boundingBoxes.count {
            if i < predictions.count {
                let prediction = predictions[i]
                
                // The predicted bounding box is in the coordinate space of the input
                // image, which is a square image of 416x416 pixels. We want to show it
                // on the video preview, which is as wide as the screen and has a 16:9
                // aspect ratio. The video preview also may be letterboxed at the top
                // and bottom.
                let width = view.bounds.width
                let height = width * 16 / 9
                let scaleX = width / CGFloat(YOLO.inputWidth)
                let scaleY = height / CGFloat(YOLO.inputHeight)
                let top = (view.bounds.height - height) / 2
                
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
//                print(color)
                //        print("show the box label and color:",label,color)
                boundingBoxes[i].show(frame: rect, label: Glabel[i], color: color)
                print("enter the show function and showing the box",i)
            } else {
                boundingBoxes[i].hide()
            }
        }
    }
}
extension ViewController: VideoCaptureDelegate {
    func videoCapture(_ capture: VideoCapture, didCaptureVideoFrame pixelBuffer: CVPixelBuffer?, timestamp: CMTime) {
        // For debugging.
        //predict(image: UIImage(named: "dog416")!); return
        
        if let pixelBuffer = pixelBuffer {
            // The semaphore will block the capture queue and drop frames when
            // Core ML can't keep up with the camera.
            semaphore.wait()
            
            // For better throughput, we want to schedule multiple prediction requests
            // in parallel. These need to be separate instances, and inflightBuffer is
            // the index of the current request.
            let inflightIndex = inflightBuffer
            inflightBuffer += 1
            if inflightBuffer >= ViewController.maxInflightBuffers {
                inflightBuffer = 0
            }
            
            if useVision {
                // This method should always be called from the same thread!
                // Ain't nobody likes race conditions and crashes.
                self.predictUsingVision(pixelBuffer: pixelBuffer, inflightIndex: inflightIndex)
            } else {
                // For better throughput, perform the prediction on a concurrent
                // background queue instead of on the serial VideoCapture queue.
                DispatchQueue.global().async {
                    print("call from here directly")
                    self.predict(pixelBuffer: pixelBuffer, inflightIndex: inflightIndex)
                }
            }
        }
    }
}

//extension UIViewController{
//    @objc func swipeAction(swipe: UISwipeGestureRecognizer) {
//        switch swipe.direction.rawValue{
//        case 1:
//            performSegue(withIdentifier: "goLeft", sender: self)
//        case 2:
//            performSegue(withIdentifier: "goRight", sender: self)
//        default:
//            break
//        }
//
//    }
//}

extension UIFont {
    // Based on: https://stackoverflow.com/questions/4713236/how-do-i-set-bold-and-italic-on-uilabel-of-iphone-ipad
    func withTraits(traits:UIFontDescriptor.SymbolicTraits...) -> UIFont {
        let descriptor = self.fontDescriptor.withSymbolicTraits(UIFontDescriptor.SymbolicTraits(traits))
        return UIFont(descriptor: descriptor!, size: 0)
    }
}
