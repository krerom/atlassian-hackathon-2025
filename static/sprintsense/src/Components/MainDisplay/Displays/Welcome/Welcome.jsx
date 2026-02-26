import React from "react";
import styles from "./Welcome.module.css";

export default function WelcomePage({ userData, setPage, showWelcome, setshowWelcome }) {
  const goToOverview = () => {
    setPage("overview");
  };

  const handleCheckboxChange = () => {
    if (!showWelcome == true) {
      localStorage.setItem("showWelcome", "true");
    } else {
      localStorage.setItem("showWelcome", "false");
    }
    setshowWelcome(!showWelcome);

  };

  return (
    <div className={styles.container}>
      <div className={styles.card}>
        <h1>Welcome to SprintSense</h1>
        <p>Your personal sprint analytics and simulation platform awaits.</p>
        <button className={styles.cta} onClick={goToOverview}>
          Overview
        </button>
        <div className={styles.checkContainer}>
          <label htmlFor="checkbox">Show Welcome Message</label>
          <input
            type="checkbox"
            id="checkbox"
            name="welcomeMsg"
            // Connect the input's 'checked' state to the React state
            checked={showWelcome}
            // Call the handler on change
            onChange={handleCheckboxChange}
          />
        </div>
      </div>
    </div>
  );
}
