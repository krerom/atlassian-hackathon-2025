import styles from "./MainDisplay.module.css"
import React, { useState, useEffect } from "react";
import OverviewDisplay from "./Displays/OverviewDisplay/OverviewDisplay";
import MetricsDisplay from "./Displays/MetricsDisplay/MetricsDisplay";
import SprintsDisplay from "./Displays/SprintsDisplay/SprintsDisplay";
import SettingsDisplay from "./Displays/SettingsDisplay/SettingsDisplay";
import FAQs from "./Displays/FAQs/FAQs";
import Simulation from "./Displays/Simulation/Simulation";
import { AnimatePresence, motion } from "framer-motion";
import WelcomePage from "./Displays/Welcome/Welcome";
import { view, invoke, requestJira } from '@forge/bridge';

function MainDisplay({ page, sprints, userData, setUserData, setPage, predictedSimSprint, setPredictedSimSprint, llmSimResponse, setLlmSimResponse, settourResKey, setshowWelcome, showWelcome }) {
    let content;
    const [metrics, setMetrics] = useState([]);
    const [priorMetrics, setPriorMetrics] = useState([]);
    

    const contentVariants = {
        hidden: { opacity: 0, y: 8 },
        show: {
            opacity: 1,
            y: 0,
            transition: {
                type: "spring",
                stiffness: 110,
                damping: 20,
                mass: 0.5
            }
        },
        exit: {
            opacity: 0,
            y: -6,
            transition: {
                duration: 0.15
            }
        }
    };

    const handleTutorialRestart = () => {
        localStorage.removeItem("menu");
        localStorage.removeItem("overview");
        localStorage.removeItem("metrics");
        localStorage.removeItem("sim");
        localStorage.removeItem("settings");
        settourResKey(prev => prev + 1);
        setPage("welcome");
    };


    switch (page) {
        case 'welcome': content = <WelcomePage userData={userData} setPage={setPage} setshowWelcome={setshowWelcome} showWelcome={showWelcome}/>; break;
        case 'overview': content = <OverviewDisplay sprints={sprints} userData={userData} setPage={setPage} />; break;
        case 'metrics': content = <MetricsDisplay sprints={sprints} userData={userData} metrics={metrics} setMetrics={setMetrics} setUserData={setUserData} priorMetrics={priorMetrics} setPriorMetrics={setPriorMetrics} />; break;
        case 'sprint-details': content = <SprintsDisplay sprints={sprints} />; break;
        case 'settings': content = <SettingsDisplay userData={userData} sprints={sprints} setshowWelcome={setshowWelcome} showWelcome={showWelcome} handleTutorialRestart={handleTutorialRestart}/>; break;
        case 'faq': content = <FAQs userData={userData} sprints={sprints} />; break;
        case 'simulation': content = <Simulation userData={userData} sprints={sprints} predictedSimSprint={predictedSimSprint} setPredictedSimSprint={setPredictedSimSprint}
            llmSimResponse={llmSimResponse} setLlmSimResponse={setLlmSimResponse} />; break;
    }

    return (
        <div className={styles.container}>
            <AnimatePresence mode="wait">
                <motion.div
                    key={page}
                    variants={contentVariants}
                    initial="hidden"
                    animate="show"
                    exit="exit"
                    style={{ height: "100%" }}
                >
                    {content}
                </motion.div>
            </AnimatePresence>
        </div>
    );
}

export default MainDisplay;