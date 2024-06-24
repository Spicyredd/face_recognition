import React, { useState } from 'react';
import Login from './components/Login';
import Dasboard from './components/Dasboard';

const App = () => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  const handleLogin = (authStatus) => {
    setIsAuthenticated(authStatus);
  };

  return (
    <div>
      {isAuthenticated ? (
        <Dasboard />
      ) : (
        <Login onLogin={handleLogin} />
      )}
    </div>
  );
};

export default App;
