import React, { useState } from 'react';
import Modal from 'react-modal';
import { BrowserRouter as Router, Route, Routes, Navigate } from 'react-router-dom';
import Login from './components/Login';
import Dasboard from './components/Dasboard'; // Updated import statement
import AttendancePage from './components/AttendancePage';
import Teacher from './components/Teacher';

Modal.setAppElement('#root');

const App = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  const handleLogin = (authStatus) => {
    setIsAuthenticated(authStatus);
  };

  return (
    <Router>
      <Routes>
        <Route
          path="/"
          element={isAuthenticated ? <Dasboard /> : <Login onLogin={handleLogin} />}
        />
        <Route
          path="/attendance"
          element={isAuthenticated ? <AttendancePage /> : <Navigate to="/" />}
        />
        <Route
          path="/teacher"
          element={isAuthenticated ? <Teacher /> : <Navigate to="/" />}
        />
      </Routes>
    </Router>
  );
};

export default App;
