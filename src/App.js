import React from 'react'
import Header from './Components/Header';
import Form from './Components/Form';
import './style.css';
import Button from './Components/Button';
import Hints from './Components/Hints';
import Imagesite from './Components/Imagesite';
import Layer1 from './Components/Layer1';
import Shadowimg from './Components/Shadowimg';
import Studentimg from './Components/Studentimg';
import Logoimg from './Components/Logoimg';
const App = () => {
  return (
    <div className="App">
      <div className='image-selector'>
          <Logoimg  />
          <Imagesite  />
          <Layer1 />
          <Shadowimg  />
          <Studentimg />
      </div>

      <div className='pannel'>
        <Header />
        <Form />
        <Hints />
        <Button />
      </div>
    </div>
  )
}

export default App