import React from 'react'

const Form = () => {
  return (
    <div className='form-container'>
      <form>
        <input className='id' type="text" placeholder="Username" />
        <input className='id' type="password" placeholder="password" />
      </form>
    </div>
  )
}

export default Form