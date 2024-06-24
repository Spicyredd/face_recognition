import React, { useState } from 'react';
import logoking from "../img/frame.png";
import Idds from "../img/idd.png";
import Lock from "../img/lock.png";
import student from "../img/image1.png";
import Layering from "../img/layer1.png";
import '../styles/login.css';  // Import the CSS file from the centralized folder

const Login = ({ onLogin }) => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (username === 'admin' && password === '123') {
      onLogin(true);
    } else {
      alert('Invalid credentials');
    }
  };

  return (
    <div className="login-container">
      <div className="login-logo">
        <img src={logoking} alt="Attendes Logo" />
      </div>
      <div className="login-box">
        <h2>Welcome to</h2>
        <h3>Smart Attendance Management System</h3>
        <p>Log in to access your account.</p>
        <form onSubmit={handleSubmit}>
          <div className="input-group">
            <img src={Idds} alt="ID Icon" />
            <input
              type="text"
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
          </div>
          <div className="input-group">
            <img src={Lock} alt="Lock Icon" />
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          <div className="login-options">
            <label>
              <input type="checkbox" />
              <span>Remember me </span>
            </label>
            <a href="#">Forget password?</a>
          </div>
          <button type="submit" className="login-btn">Log in</button>
        </form>
      </div>
      <div className="login-image">
        <img src={Layering} alt="" className='layering_pic' />
        <img src={student} alt="Student" className='student_img' />
      </div>
    </div>
  );
};

export default Login;
