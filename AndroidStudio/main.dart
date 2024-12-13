import 'package:flutter/material.dart';
import 'API.dart';
import 'dart:async';
import 'dart:convert';
import 'package:image/image.dart' as img;
import 'dart:io';
import 'package:flutter/material.dart';

dynamic url;
dynamic url2;
String link = 'http://192.168.219.101:5000';
int t = 1;

void main(){
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return const MaterialApp(
      home: FirstPage(),
    );
  }
}

class FirstPage extends StatefulWidget {
  const FirstPage({super.key});

  @override
  State<FirstPage> createState() => _FirstPageState();
}

class _FirstPageState extends State<FirstPage> {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        elevation: 0.0,
        title: Text('게임 AI',style: TextStyle(color: Colors.black),),
        backgroundColor: Colors.white54,
      ),
      body: Center(

        child: ElevatedButton(
          onPressed: (){
            gget(link + '/6');
            url = link + '/0';
            Navigator.push(context, MaterialPageRoute(builder: (context)=>SecondPage()));},
          child: Text('Start',style: TextStyle(color: Colors.black),),
          style: ElevatedButton.styleFrom(
              primary: Colors.grey[100]
          ),
        ),
      ),
      backgroundColor: Colors.white70,
    );
  }
}

class SecondPage extends StatefulWidget {
  const SecondPage({super.key});

  @override
  State<SecondPage> createState() => _SecondPageState();
}

class _SecondPageState extends State<SecondPage> {
  dynamic data;
  late Timer _timer;

  void initState() {
    super.initState();
    _timer = Timer.periodic(
      const Duration(milliseconds: 160),
          (_) => _incrementCounter(),
    );
  }

  void _incrementCounter() {
    setState(() {
      t = t+1;
      if(t%2==0){
        url2 = link + '/static/images/img.png/?t=${t}';}
      else{
        gget(url);
        if(url == link + '/4'){
          url = link + '/5';
        }
      }
    }
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white54,
      appBar: AppBar(
        backgroundColor: Colors.white70,
        title: Text('게임',style: TextStyle(color: Colors.black),),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            Row(mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Image.network(
                  url2,
                  height:400,
                  width: 400,
                ),
              ]
            ),
            Row(mainAxisAlignment: MainAxisAlignment.center,
              children: [
                TextButton(
                    onPressed: () {
                      url = link + '/1';
                    },
                    child: Text('fire',style: TextStyle(color: Colors.black))
                ),
                TextButton(
                    onPressed: (){
                      url = link + '/3';
                    },
                    child: Text('<-',style: TextStyle(color: Colors.black))
                ),
                TextButton(
                    onPressed: (){
                      url = link + '/0';
                    },
                    child: Text('stop',style: TextStyle(color: Colors.black))
                ),
                TextButton(
                    onPressed: () {
                      url = link + '/2';
                    },
                    child: Text('->',style: TextStyle(color: Colors.black))
                ),
                TextButton(
                    onPressed: (){
                      url = link + '/4';},
                    child: Text('end',style: TextStyle(color: Colors.black))
                ),
              ],
            ),
            TextButton(
                onPressed: () {
                },
                child: Text(link,style: TextStyle(color: Colors.black))
            ),
          ],
        ),
      ),
    );
  }
}