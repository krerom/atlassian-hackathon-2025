import React, { createContext, useState, useContext } from "react";
import Joyride, { ACTIONS, EVENTS, STATUS } from "react-joyride";

const TourContext = createContext();
export const useTour = () => useContext(TourContext);

export default function TourProvider({ children }) {
  const [run, setRun] = useState(false);
  const [steps, setSteps] = useState([]);
  const [storageKey, setStorageKey] = useState("sprintSenseTourDone");

  function startTour(pageSteps, pageKey = "sprintSenseTourDone") {
    setSteps(pageSteps);
    setStorageKey(pageKey);
    setRun(true);
  }

  return (
    <TourContext.Provider value={{ startTour }}>
      {children}

      <Joyride
        steps={steps}
        run={run}
        continuous
        showSkipButton
        callback={(data) => {
          const { status } = data;
          if ([STATUS.FINISHED, STATUS.SKIPPED].includes(status)) {
            // Reset run so next page tour can start
            localStorage.setItem(storageKey, "true");
            setRun(false);
          }
        }}
      />
    </TourContext.Provider>
  );
}
