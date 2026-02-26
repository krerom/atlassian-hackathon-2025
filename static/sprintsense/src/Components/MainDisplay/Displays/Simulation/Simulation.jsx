import styles from "./Simulation.module.css"
import React, { useState, useEffect } from "react"
import { TrendingUp, Calendar, Target, AlertCircle, Play, Edit2, Lock, ChevronRight } from 'lucide-react';
import { invoke } from '@forge/bridge';
import Loader from "../../../Loader/Loader.jsx";
import ProgressBar from "../../../ProgressBar/ProgressBar.jsx";
import { useTour } from "../../../../utils/TourProvider.jsx";

function Simulation({ userData, sprints, predictedSimSprint, setPredictedSimSprint, llmSimResponse, setLlmSimResponse }) {
    const { startTour } = useTour();
    const [predicting, setPredicting] = useState(false);
    const [displayedSprints, setDisplayedSprints] = useState([]);
    const [simInputs, setSimInputs] = useState({
        sprint_id: "",
        number_of_issues: "",
        team_size: "",
        sprint_duration_days: "",
        completed_issues_prev_sprint: "",
        velocity_prev_sprint: "",
        avg_story_points_per_member: ""
    });

    let timeSortedSprints;

    useEffect(() => {
        let sprintsToDisplay;
        if (sprints?.length > 0) {
            const allSprints = sprints?.flatMap(board => board.sprints);

            timeSortedSprints = allSprints?.sort((a, b) => {
                const dateA = new Date(a.startDate);
                const dateB = new Date(b.startDate);

                return dateA - dateB;
            });
            sprintsToDisplay = timeSortedSprints?.slice(-6);
        } else {
            // create synthetic sprints here
            // id state name
            sprintsToDisplay = [
                { id: 0, state: 'closed', name: "Dummy Sprint 1" },
                { id: 1, state: 'closed', name: "Dummy Sprint 2" },
                { id: 2, state: 'closed', name: "Dummy Sprint 3" },
                { id: 3, state: 'ongoing', name: "Dummy Sprint 4" },
                { id: 4, state: 'ongoing', name: "Dummy Sprint 5" }
            ];
        }

        setDisplayedSprints(sprintsToDisplay);

        console.log("displayedSprints: ", sprintsToDisplay);

        const simSprintID = sprintsToDisplay?.at(-1).id + 1;
        setSimInputs(prev => ({ ...prev, sprint_id: simSprintID }));
        if (!localStorage.getItem("sim")) {
            startTour(
                [
                    {
                        target: "#sim-hist",
                        content: (
                            <>
                                <strong>Historical Sprints:</strong><br />
                                Review your past sprints here. You can modify ongoing sprints,
                                but completed ones remain closed and cannot be changed.
                            </>
                        ),
                    },
                    {
                        target: "#sim-simsprint",
                        content: (
                            <>
                                <strong>Planned Sprint:</strong><br />
                                Adjust the parameters of your upcoming sprint and click <strong>'Simulate'</strong>
                                to see predicted metrics and outcomes.
                            </>
                        ),
                    },
                    {
                        target: "#sim-metrcs",
                        content: (
                            <>
                                <strong>Predicted Metrics:</strong><br />
                                Based on your input, these metrics estimate the likely sprint outcome.
                                Will it finish on time? How long could it take? How much work will be completed?
                            </>
                        ),
                    },
                    {
                        target: "#sim-llm",
                        content: (
                            <>
                                <strong>Natural Language Insights:</strong><br />
                                Here you can see a natural language explanation of predictions and metrics,
                                helping you understand and interpret the results.
                            </>
                        ),
                    }
                ], "sim"
            );
        }
    }, []);

    useEffect(() => {
        setSimInputs(prev => ({ ...prev, avg_story_points_per_member: Number(simInputs.number_of_issues) / Number(simInputs.team_size) }));
        console.log(Number(simInputs.number_of_issues) / Number(simInputs.team_size));
    }, [simInputs.number_of_issues, simInputs.team_size]);

    const handleSimPrediction = async () => {
        // pass actual sprints as they come - they will be transformed to features in backend
        // create sim sprint already as features object

        // completed_issues_prev_sprint and velocity_prev_sprint are available in the backend after 
        // generating features for regular sprints, set them there
        if (predicting) return;
        setPredicting(true);

        try {
            const sim_prediction = await invoke("generateLSTMsimulation", { auth_token: userData?.auth_token, sprints: displayedSprints, sim_sprint: simInputs });
            console.log(sim_prediction);
            if (sim_prediction.success == true) {
                setPredictedSimSprint(JSON.parse(sim_prediction.sprint));

                const llmResponse = await invoke("generateLLMinference", { auth_token: userData?.auth_token, sprints: timeSortedSprints, predicted_sprint: sim_prediction.sprint });
                if (llmResponse) {
                    setLlmSimResponse(llmResponse);
                }
            }
        } catch (error) {
            console.log(`Error generating simulated prediction: ${error}`);
        } finally {
            setPredicting(false);
        }

    };

    return (
        <>
            <div className={styles.container}>
                <div className={styles.heading}>
                    <h1>Simulation Mode</h1>
                    <p>Plan and predict your next sprint based on historical data</p>
                </div>

                <div className={styles.historicPrintsCont} id="sim-hist">
                    <div className={styles.subtitle}>
                        <span><Calendar /></span>
                        <span>Historical Sprints</span>
                    </div>
                    <div className={styles.cardsCont}>
                        {displayedSprints?.map((sprint, index) => (
                            <>
                                <div key={sprint?.id} className={`${styles.sprintCard} ${sprint.state != "closed" ? styles.active : styles.inactive}`}>
                                    <span>{sprint?.name}</span>
                                    <span>{sprint?.state}</span>
                                </div>
                                {index < displayedSprints?.length - 1 && <ChevronRight />}
                            </>
                        ))}
                    </div>
                </div>

                <div className={styles.simulatedSprintCont} id="sim-simsprint">
                    <div className={styles.subtitle}>
                        <span><Target /></span>
                        <span>New Sprint Simulation</span>
                    </div>
                    <div className={styles.simulatedSprint}>
                        <div className={styles.simSprintHead}>
                            <p>Configure your next sprint parameters.</p>
                            {predicting ? <span><Loader /></span> : <span onClick={handleSimPrediction}><Play />Simulate Sprint</span>}
                        </div>
                        <div className={styles.simSprintInputs}>
                            <div className={styles.simSprintInput}>
                                <label>Story Points</label>
                                <input type="number"
                                    value={simInputs.storyPoints}
                                    onChange={(e) => setSimInputs(prev => ({ ...prev, number_of_issues: e.target.value }))} />
                            </div>
                            <div className={styles.simSprintInput}>
                                <label>Team Capaity</label>
                                <input type="number"
                                    value={simInputs.teamSize}
                                    onChange={(e) => setSimInputs(prev => ({ ...prev, team_size: e.target.value }))} />
                            </div>
                            <div className={styles.simSprintInput}>
                                <label>Sprint Duration (days)</label>
                                <input type="number"
                                    value={simInputs.durationDays}
                                    onChange={(e) => setSimInputs(prev => ({ ...prev, sprint_duration_days: e.target.value }))} />
                            </div>
                        </div>
                    </div>
                </div>

                {predictedSimSprint &&
                    <div className={styles.predictionsCont}>
                        <div className={styles.subtitle}>
                            <span><TrendingUp /></span>
                            <span>Simulation Results</span>
                        </div>
                        <div className={styles.metricsContainer} id="sim-metrcs">
                            <div className={styles.compareBox}>
                                <span className={styles.compareTitle}>Velocity</span>
                                <span className={styles.compareValue}>{predictedSimSprint?.predictedVelocity}</span>
                            </div>

                            <div className={styles.compareBox}>
                                <span className={styles.compareTitle}>Predicted Duration (Days)</span>
                                <span className={styles.compareValue}>{predictedSimSprint?.predictedDuration}</span>
                            </div>

                            <div className={styles.compareBox}>
                                <span className={styles.compareTitle}>Completed on Time</span>
                                <span className={styles.compareValue}>{predictedSimSprint?.predictedFinishedOnTime == 1 ? "Yes" : "No"}</span>
                            </div>
                        </div>
                        {llmSimResponse && <div id="sim-llm" className={styles.llmContainer} dangerouslySetInnerHTML={{ __html: llmSimResponse }}></div>}
                    </div>
                }
            </div>
        </>
    );
}

export default Simulation;