import React, { useState } from 'react';
import Modal from 'react-modal';
import '../styles/Teacher.css';

const classes = [
  { id: 1, className: 'Class A', subject: 'Math', time: '10:00 AM' },
  { id: 2, className: 'Class B', subject: 'Science', time: '11:00 AM' },
];

const Teacher = () => {
  const teacherName = "Mr. Smith"; 
  const todayDate = new Date().toLocaleDateString();

  const [modalIsOpen, setModalIsOpen] = useState(false);
  const [selectedClass, setSelectedClass] = useState(null);
  const [newTime, setNewTime] = useState('10:00');

  const openModal = (classItem) => {
    setSelectedClass(classItem);
    setNewTime(classItem.time); 
    setModalIsOpen(true);
  };

  const closeModal = () => {
    setModalIsOpen(false);
    setSelectedClass(null);
  };

  const handlePostpone = async () => {
    // Construct the data to send to the backend
    const updatedClass = {
      id: selectedClass.id,
      time: newTime
    };
    console.log(updatedClass.time)
    try {
      // Replace with your actual backend API endpoint
      const response = await fetch('http://localhost:3000', { // Replace with your actual endpoint
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updatedClass)
      });

      if (response.ok) {
        console.log('Class time updated successfully!');
        // Optionally, you can refresh the class list here
      } else {
        console.error('Error updating class time');
      }

      closeModal();
    } catch (error) {
      console.error('Error:', error);
    }
  };

  return (
    <div className="teacher-container">
      <h1>Teacher Page</h1>
      <div className="teacher-info">
        <p><strong>Teacher Name:</strong> {teacherName}</p>
        <p><strong>Today's Date:</strong> {todayDate}</p>
      </div>
      <div className="classes-today">
        <h2>Classes Today</h2>
        {classes.map((classItem) => (
          <div key={classItem.id} className="class-item">
            <p><strong>Class Name:</strong> {classItem.className}</p>
            <p><strong>Subject:</strong> {classItem.subject}</p>
            <p><strong>Time:</strong> {classItem.time}</p>
            <button onClick={() => openModal(classItem)}>Postpone Time</button>
          </div>
        ))}
      </div>

      {selectedClass && (
        <Modal
          isOpen={modalIsOpen}
          onRequestClose={closeModal}
          contentLabel="Postpone Time Modal"
          className="modal"
          overlayClassName="overlay"
        >
          <div className="modal-content">
            <h2>Postpone Time for {selectedClass.className}</h2>
            <div className="time-picker">
              <input type="number" min="0" max="23" value={newTime.slice(0, 2)} onChange={(e) => setNewTime(`${e.target.value}:${newTime.slice(3, 5)}`)} />
              <span>:</span>
              <input type="number" min="0" max="59" value={newTime.slice(3, 5)} onChange={(e) => setNewTime(`${newTime.slice(0, 2)}:${e.target.value}`)} />
            </div>
            <div className="modal-buttons">
              <button onClick={handlePostpone} className="save-btn">Save</button>
              <button onClick={closeModal} className="cancel-btn">Cancel</button>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
};

export default Teacher;