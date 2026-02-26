import styles from "./FAQs.module.css"
import React, {useState, useEffect} from "react"

function FAQs(){
    const [activeId, setActiveId] = useState(null);

    const faqs = [
        {id: 1, title: "How do I start generating predictions?", text: "First, secure your access: Go to 'Settings' -> 'LLM Settings' and input your OpenAI API Key. Second, personalize the model: Navigate to 'Settings' -> 'LSTM Settings' and click 'Train Adapter' to personalize the model with your team's history."},
        {id: 2, title: "Why is a minimum of 6 sprints required?", text: "Our LSTM (Long Short-Term Memory) model requires a minimum of 6 historical sprints to establish stable trends and sufficient context for accurate forecasting. This minimum is crucial for the model's performance."},
        {id: 3, title: "How should I interpret the charts?", text: "Click on any Chart Title or the Metric Name within the dashboard to open a detailed explanation of the calculation method (e.g., EMA-6) and how the metric relates to your predicted outcome."},
        {id: 4, title: "What is the 'Adapter' and why train it?", text: "The adapter is a small, specialized layer of the LSTM model that is trained exclusively on your team's historical data. This personalization step ensures the general machine learning model generates predictions that reflect your specific velocity patterns, team size, and sprint cadence."},
        {id: 5, title: "What does the 'Predicted Finished On Time' metric mean?", text: "This metric provides the model's forecast (as a probability, usually 1 or 0) that your next committed sprint will be completed by the planned end date. A value of 1 suggests the commitment is realistic based on historical performance and current features."},
        {id: 6, title: "I see the error 'Please add API Key.' What should I do?", text: "This indicates your system is missing the required OpenAI API Key needed to run the LLM analysis (the text interpretation of your data). Please go to 'Settings' -> 'LLM Settings' to enter and save your key."},
        {id: 7, title: "What is the difference between Velocity and Completed Issues?", text: "Velocity typically refers to the sum of Story Points completed in a sprint. Our charts track the Total Completed Story Points (your team's true velocity) and compare it against the Number of Issues Completed to help identify consistency in story point estimation."},
        {id: 8, title: "Why can't I edit closed sprints in Simulation Mode?", text: "Simulation Mode is designed for time series forecasting to predict the outcome of future sprints. Editing closed sprints would corrupt the historical data required for accurate modeling, as these sprints represent an immutable, completed record of past work. You can only adapt elements (like scope or capacity) of sprints that are currently ongoing or planned for the future, as these are the only elements that can realistically interfere with your future workflow."}
    ];

    const handleClick = (id) => {
        setActiveId(prevId => (prevId === id ? null : id));
    };

    return(
        <div className={styles.container}>
            {faqs.map((faq)=>(
                <div key={faq.id} className={styles.faqContainer}>
                    <span className={styles.faqTitle} onClick={() => handleClick(faq.id)}>{faq.title}</span>
                    {activeId === faq.id && (
                        <div className={styles.faqTextWrapper}>
                            {/* Render the text content only if the ID matches the active state */}
                            <span className={styles.faqText}>
                                {faq.text}
                            </span>
                        </div>
                    )}
                </div>
            ))}
        </div>
    );
}

export default FAQs;