import React from 'react';
import { Link } from 'react-router-dom';

const Dashboard = () => {
  return (
    <div>
      <h1>Dashboard</h1>
      <ul>
        <li>
          <Link to="/attendance">Attendance</Link>
        </li>
        <li>
          <Link to="/teacher">Teacher</Link>
        </li>
      </ul>
    </div>
  );
};

export default Dashboard;
