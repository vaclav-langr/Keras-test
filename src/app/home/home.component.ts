import { Component, OnInit } from '@angular/core';
import * as KerasJS from 'keras-js';

@Component({
  selector: 'my-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {

  constructor() {
    // Do stuff
  }

  ngOnInit() {
    var model = new KerasJS.Model();
    console.log('Hello Home');
  }

}
