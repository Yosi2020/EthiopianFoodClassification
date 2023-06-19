import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'dart:io';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:tflite_flutter_helper/tflite_flutter_helper.dart';
import 'package:image/image.dart' as img;

void main() {
  runApp(MaterialApp(home: MyApp(),
    routes: {
      '/about': (context) => AboutUsScreen(),
      '/info': (context) => InfoScreen(),
    },
  ));
}

class MyApp extends StatefulWidget {
  @override
  _MyAppState createState() => _MyAppState();
}

class _MyAppState extends State<MyApp> {
  File _image;
  String _label = '';
  Interpreter _interpreter;
  List<String> _labels = ["Beyaynetu (በያይነቱ)", "Chechebsa (ጨጨብሳ)",
    "Doro Wat (ዶሮ ወጥ)", "Fir-fir (ፍርፍር)", "Genfo (ገንፎ)",
    "Kikil (ቅቅል)", "Kitfo (ክትፎ)", "Shekla Tibs (ሸክላ ጥብስ)",
    "Shiro Wat (ሽሮ ወጥ)", "Tihlo (ጥህሎ)", "Tire Siga(ጥሬ ስጋ)"]; // replace with your labels
  final int _inputSize = 224; // replace with the input size of your model
  ImageProcessor imageProcessor;
  /// Shapes of output tensors
  List<List<int>> _outputShapes;

  /// Types of output tensors
  List<TfLiteType> _outputTypes;
  bool isPredict = false;
  bool isImage = false;

  @override
  void initState() {
    super.initState();
    loadModel();
  }


  String modelFileName = "model/model_E_Food.tflite";  // Replace with your model file name

  Future<void> loadModel({Interpreter interpreter}) async {
    try {
      _interpreter = interpreter ??
          await Interpreter.fromAsset(
            modelFileName,
            options: InterpreterOptions()..threads = 4, //myOptions,
          );

      var outputTensors = _interpreter.getOutputTensors();
      // print("the length of the ouput Tensors is ${outputTensors.length}");
      _outputShapes = [];
      _outputTypes = [];
      outputTensors.forEach((tensor) {
        print(tensor.toString());
        _outputShapes.add(tensor.shape);
        _outputTypes.add(tensor.type);
      });
    } catch (e) {
      print("Error while creating interpreter: $e");
    }
  }


  Future getImage(bool isCamera) async {
    final image = await ImagePicker().getImage(
        source: isCamera ? ImageSource.camera : ImageSource.gallery
    );

    setState(() {
      _image = File(image.path);
      isImage = true;
      isPredict = false;
    });
  }

  Future<void> classifyImage(File Img) async {
    if (_interpreter == null) {
      print("Interpreter not initialized.");
      return;
    }

    // Load image and preprocess it
    Uint8List imageBytes = await Img.readAsBytes();
    img.Image rawImage = img.decodeImage(imageBytes);
    img.Image resizedImage = img.copyResize(rawImage, width: 224, height: 224);

    // Normalize pixel values and convert image to tensor format
    var input = List<List<List<List<double>>>>.generate(
      1,
          (_) => List<List<List<double>>>.generate(
        224,
            (_) => List<List<double>>.generate(
          224,
              (_) => List<double>.generate(3, (_) => 0),
        ),
      ),
    );

    for (var y = 0; y < resizedImage.height; y++) {
      for (var x = 0; x < resizedImage.width; x++) {
        var pixel = resizedImage.getPixel(x, y);
        var r = img.getRed(pixel) / 255.0;
        var g = img.getGreen(pixel) / 255.0;
        var b = img.getBlue(pixel) / 255.0;

        input[0][y][x][0] = r;
        input[0][y][x][1] = g;
        input[0][y][x][2] = b;
      }
    }

    // Run model
    var output = TensorBuffer.createFixedSize(<int>[1, 10], TfLiteType.float32);
    _interpreter.run(input, output.buffer);

    // Interpret output
    var outputList = output.getDoubleList();
    var maxConfidence = outputList.reduce((curr, next) => curr > next ? curr : next);
    var predictedClass = outputList.indexOf(maxConfidence);

    print('Predicted class: $predictedClass with confidence $maxConfidence');
    setState(() {
      _label = maxConfidence > 0.5 ? _labels[predictedClass] : 'Unknown';
      isPredict = true;
      isImage = false;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Ethiopian Food Classifier'),
        centerTitle: true,
        backgroundColor: Colors.teal,
        actions: [
          PopupMenuButton<String>(
            onSelected: (value) {
              if (value == 'about') {
                Navigator.pushNamed(context, '/about');
              } else if (value == 'info') {
                Navigator.pushNamed(context, '/info');
              }
            },
            itemBuilder: (BuildContext context) {
              return [
                PopupMenuItem(
                  value: 'about',
                  child: Text('About Us'),
                ),
                PopupMenuItem(
                  value: 'info',
                  child: Text('Information'),
                ),
              ];
            },
          ),
        ],
      ),
      body: Container(
        decoration: BoxDecoration(
          gradient: LinearGradient(
            colors: [Colors.teal, Colors.blue], // Replace with your desired colors
            begin: Alignment.topCenter,
            end: Alignment.bottomCenter,
          ),
        ),
        child: Center(
          child: Padding(
            padding: const EdgeInsets.only(left:0, top:0, right:0, bottom: 60),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: <Widget>[
                _image == null
                    ? Text(
                  'No image selected...',
                  style: TextStyle(fontSize: 18.0),
                )
                    : Container(
                  width: MediaQuery.of(context).size.width*0.98,
                  height: MediaQuery.of(context).size.height*0.67,
                  child: InteractiveViewer(
                    boundaryMargin: EdgeInsets.all(20.0),
                    minScale: 0.1,
                    maxScale: 5,
                    child: Container(
                      decoration: BoxDecoration(
                        image: DecorationImage(
                          image: FileImage(_image),
                          fit: BoxFit.cover,
                        ),
                        borderRadius: BorderRadius.all(Radius.circular(15)),
                        border: Border.all(
                          color: Colors.blueAccent,
                          width: 2,
                        ),
                      ),
                    ),
                  ),
                ),
                SizedBox(height: 20),
                isPredict?
                  Text(
                    'Label: $_label',
                    style: TextStyle(fontSize: 30.0, fontWeight: FontWeight.bold),
                  ) : isImage ? TextButton(
                  onPressed: () {
                    setState(() {
                    });
                    classifyImage(_image);
                  },
                  child: Text(
                    "Predict",
                    style: TextStyle(color: Colors.white, fontSize: 20),
                  ),
                  style: ButtonStyle(
                    backgroundColor: MaterialStateProperty.all(Colors.teal),
                  ),
                ) : Container()
              ],
            ),
          ),
        ),
      ),
      floatingActionButton: Row(
        mainAxisAlignment: MainAxisAlignment.end,
        children: <Widget>[
          FloatingActionButton(
            heroTag: 'camera',
            onPressed: () => getImage(true),
            tooltip: 'Pick Image from Camera',
            child: Icon(Icons.camera),
            backgroundColor: Colors.teal,
          ),
          SizedBox(width: 10),
          FloatingActionButton(
            heroTag: 'gallery',
            onPressed: () => getImage(false),
            tooltip: 'Pick Image from Gallery',
            child: Icon(Icons.photo_library),
            backgroundColor: Colors.teal,
          ),
        ],
      ),
    );
  }
}




class InfoScreen extends StatelessWidget {
  final List<String> foodClasses = [
    "Beyaynetu (በያይነቱ)",
    "Chechebsa (ጨጨብሳ)",
    "Doro Wat (ዶሮ ወጥ)",
    "Fir-fir (ፍርፍር)",
    "Genfo (ገንፎ)",
    "Kikil (ቅቅል)",
    "Kitfo (ክትፎ)",
    "Shekla Tibs (ሸክላ ጥብስ)",
    "Shiro Wat (ሽሮ ወጥ)",
    "Tihlo (ጥህሎ)",
    "Tire Siga(ጥሬ ስጋ)"
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Model Information'),
        backgroundColor: Colors.teal,
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'About the Model:',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 10),
            Text(
              'This model is developed to classify Ethiopian food types.',
              style: TextStyle(fontSize: 16),
            ),
            SizedBox(height: 20),
            Text(
              'Food Classes:',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 10),
            ListView.builder(
              shrinkWrap: true,
              physics: NeverScrollableScrollPhysics(),
              itemCount: foodClasses.length,
              itemBuilder: (context, index) {
                return ListTile(
                  title: Text(foodClasses[index]),
                );
              },
            ),
            SizedBox(height: 20),
            Text(
              'Usage Instructions:',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 10),
            Text(
              '1. Take or select an image of Ethiopian food.',
              style: TextStyle(fontSize: 16),
            ),
            SizedBox(height: 5),
            Text(
              '2. Tap the "Classify" button to classify the food.',
              style: TextStyle(fontSize: 16),
            ),
            SizedBox(height: 5),
            Text(
              '3. The predicted food type will be displayed.',
              style: TextStyle(fontSize: 16),
            ),
          ],
        ),
      ),
    );
  }
}


class AboutUsScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('About Us'),
        backgroundColor: Colors.teal,
      ),
      body: Padding(
        padding: const EdgeInsets.all(20.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'About the Project:',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 10),
            Text(
              'This project is developed to classify Ethiopian food types using machine learning techniques.',
              style: TextStyle(fontSize: 16),
            ),
            SizedBox(height: 20),
            Text(
              'Developer:',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            SizedBox(height: 10),
            Text(
              'Eyosiyas Tibebu',
              style: TextStyle(fontSize: 16),
            ),
            SizedBox(height: 5),
            Text(
              'Email: eyosiyas.tibebuendalamaw@gmail.com',
              style: TextStyle(fontSize: 16),
            ),
            SizedBox(height: 5),
            Text(
              'Website: http://eyosiyastibebu.com/',
              style: TextStyle(fontSize: 16),
            ),
          ],
        ),
      ),
    );
  }
}



































// import 'dart:io';
// import 'package:flutter/material.dart';
// import 'package:image_picker/image_picker.dart';
// import 'package:flutter_tflite/flutter_tflite.dart';
//
// void main() => runApp(MyApp());
//
// class MyApp extends StatelessWidget {
//   @override
//   Widget build(BuildContext context) {
//     return MaterialApp(
//       home: Home(),
//     );
//   }
// }
//
// class Home extends StatefulWidget {
//   @override
//   _HomeState createState() => _HomeState();
// }
//
// class _HomeState extends State<Home> {
//   File? _image;
//   List? _result; // Initialize _result as null
//
//   @override
//   void initState() {
//     super.initState();
//     loadModel().then((value) {
//       setState(() {});
//     });
//   }
//
//   loadModel() async {
//     String? res;
//     try {
//       res = await Tflite.loadModel(
//         model: "assets/model/model_E_Food.tflite",
//         labels: "assets/model/label.txt",
//       );
//     } on Exception {
//       print('Failed to load model');
//     }
//   }
//
//   pickImage() async {
//     var image = await ImagePicker().getImage(source: ImageSource.gallery);
//     if (image == null) return;
//     setState(() {
//       _image = File(image.path);
//     });
//     classifyImage(_image!);
//   }
//
//   classifyImage(File image) async {
//     var output = await Tflite.runModelOnImage(
//       path: image.path,
//       numResults: 2,
//       threshold: 0.2,
//       imageMean: 127.5,
//       imageStd: 127.5,
//     );
//     setState(() {
//       _result = output; // Assign output, it may be null
//     });
//   }
//
//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       appBar: AppBar(
//         title: Text('Ethiopian Food'),
//       ),
//       body: Column(
//         children: <Widget>[
//           if (_image != null)
//             Container(
//               height: 250,
//               child: Image.file(_image!),
//             ),
//           Container(
//             child: _result != null // Check if _result is not null before accessing its elements
//                 ? Text(
//               _result![0]["label"], // Safely access label and provide a default value in case it's null
//             )
//                 : Text("no image is found"),
//           ),
//         ],
//       ),
//       floatingActionButton: FloatingActionButton(
//         onPressed: pickImage,
//         child: Icon(Icons.image),
//       ),
//     );
//   }
// }
