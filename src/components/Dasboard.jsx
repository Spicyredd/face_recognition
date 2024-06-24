import React from 'react';
import logo_das from '../img/logo_das.png';
import datee from '../img/date.png';
import manjil from '../img/Manjil.jpeg';
import birat from '../img/Birat.jpg';
import rishav from '../img/Rishav.jpg';
import sovit from '../img/Sovit.jpg';
import srijana from '../img/Srijana.jpg';
import shankar from '../img/Shankar.jpeg';
import sadikshya from '../img/Sadikshya.jpg';
import '../styles/Dasboard.css';

const students = [
  { id: 1, studentId: '2072-0221', name: 'Birat Gautam', class: 'Sem 1 A2', status: 'Absent', img: birat },
  { id: 2, studentId: '2072-0221', name: 'Manjil Budathoki', class: 'Sem 2 A1', status: 'Absent', img: manjil },
  { id: 3, studentId: '2072-0221', name: 'Rishav Bejjukchen', class: 'Sem 3 A2', status: 'Absent', img: rishav },
  { id: 4, studentId: '2072-0221', name: 'Sovit Kharel', class: 'Sem 8 A2', status: 'Absent', img: sovit },
  { id: 5, studentId: '2072-0221', name: 'Srijana Subedi', class: 'Sem 7 A2', status: 'Absent', img: srijana },
  { id: 6, studentId: '2072-0221', name: 'Shankar Tamang', class: 'Sem 1 A2', status: 'Absent', img: shankar},
  { id: 7, studentId: '2072-0221', name: 'Sadikshya Ghimire', class: 'Sem 1 A2', status: 'Absent', img: sadikshya}
];

function Dashboard() {
  return (
    <div className='dashboard_body'>
      <div className="dashboard-logo">
        <img src={logo_das} alt="Dashboard-Logo" />
      </div>

      <div className="dashboard-layout">
        <div className="dashboard-heading">
          <div className="date-day">
            <img src={datee} alt="date-logo" />
            <h3>Monday, June 24, 2024</h3>
          </div>
          <p>Class A1, Bsc. Hons with Artificial Intelligence</p>
        </div>

        <div className="dashboard-data-table">
          <table>
            <thead>
              <tr>
                <th>SN</th>
                <th>Student ID</th>
                <th>Student Name</th>
                <th>Class</th>
                <th>Attendance Status</th>
              </tr>
            </thead>
            <tbody>
              {students.map((student, index) => (
                <tr key={student.id}>
                  <td>{index + 1}</td>
                  <td>{student.studentId}</td>
                  <td>
                    <img src={student.img} alt={student.name} className="student-img-pic" />
                    {student.name}
                  </td>
                  <td>{student.class}</td>
                  <td className="status-absent">{student.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="pagination">
            <button className="prev">Previous</button>
            <button className="next">Next</button>
          </div>
          <button className="export-btn">Export</button>
        </div>
      </div>
    </div>
  );
}

export default Dashboard;
