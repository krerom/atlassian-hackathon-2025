import { useState, useEffect } from 'react';
import styles from './OverviewDisplay.module.css';
import { useTour } from '../../../../utils/TourProvider';

function OverviewDisplay({ userData, setPage }) {
    const { startTour } = useTour();
    const [isModelTrained, setIsModelTrained] = useState(false);
    const [hasApiKey, setHasApiKey] = useState(false);
    const [hasPredictions, setHasPredictions] = useState(false);
    const [predictionAge, setPredictionAge] = useState(null);
    const [isInsightExpanded, setIsInsightExpanded] = useState(false);
    const PREVIEW_LENGTH = 150;

    useEffect(() => {
        if (!localStorage.getItem("overview")) {
            startTour(
                [
                    {
                        target: "#overview-sys-status",
                        content: (
                            <>
                                <strong>System Status Overview:</strong><br />
                                All categories should be checked and show green.
                                Make sure to add your <strong>API Key</strong>, train your Adapter, and create predictions.
                            </>
                        ),
                    },
                    {
                        target: "#overview-sprint-prediction",
                        content: (
                            <>
                                <strong>Sprint Predictions:</strong><br />
                                This section shows <strong>LSTM predictions</strong> for your <em>next sprint</em> (which does not exist yet).
                                The model analyzes your past sprints to predict your most likely design choices for the upcoming sprint.
                            </>
                        ),
                    },
                    {
                        target: "#overview-llm",
                        content: (
                            <>
                                <strong>Natural Language Insights:</strong><br />
                                Here, the LSTM predictions along with metrics computed from your past performance
                                are translated into natural language using a Large Language Model (e.g., OpenAI ChatGPT).
                            </>
                        ),
                    }
                ]
                ,
                "overview"
            );
        }
    }, []);

    useEffect(() => {
        if (userData) {
            setIsModelTrained(!!userData.adapter_path && !!userData.adapter_created_at);
            setHasApiKey(!!userData.api_key);
            setHasPredictions(userData.prediction_created_at ? true : false);

            if (userData.prediction_created_at) {
                const age = Math.floor(
                    (Date.now() - new Date(userData.prediction_created_at).getTime()) / (1000 * 60 * 60 * 24)
                );
                setPredictionAge(age);
            }
        }
    }, [userData]);

    const getStatusIcon = (status) => {
        return status ? '✓' : '✗';
    };

    const getStatusClass = (status) => {
        return status ? styles.statusGood : styles.statusBad;
    };

    const formatDate = (dateString) => {
        if (!dateString) return 'N/A';
        return new Date(dateString).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    };

    return (
        <div className={styles.container}>
            <p className={styles.tagline}>
                The only sprint analysis tool that predicts problems BEFORE they happen and tells you exactly what to do about it.
            </p>
            {/* System Status Section */}
            <div className={styles.section} id='overview-sys-status'>
                <h2 className={styles.sectionTitle}>System Status</h2>
                <div className={styles.statusGrid} onClick={() => { setPage("settings") }}>
                    <div className={`${styles.statusCard} ${getStatusClass(hasApiKey)}`}>
                        <span className={styles.statusIcon}>{getStatusIcon(hasApiKey)}</span>
                        <div className={styles.statusContent}>
                            <h3>API Key</h3>
                            <p>{hasApiKey ? 'Configured' : 'Not Set'}</p>
                        </div>
                    </div>

                    <div className={`${styles.statusCard} ${getStatusClass(isModelTrained)}`}>
                        <span className={styles.statusIcon}>{getStatusIcon(isModelTrained)}</span>
                        <div className={styles.statusContent}>
                            <h3>LSTM Model</h3>
                            <p>{isModelTrained ? 'Trained' : 'Not Trained'}</p>
                            {isModelTrained && (
                                <span className={styles.statusDetail}>
                                    {formatDate(userData.adapter_created_at)}
                                </span>
                            )}
                        </div>
                    </div>

                    <div className={`${styles.statusCard} ${getStatusClass(hasPredictions)}`}>
                        <span className={styles.statusIcon}>{getStatusIcon(hasPredictions)}</span>
                        <div className={styles.statusContent}>
                            <h3>Predictions</h3>
                            <p>{hasPredictions ? 'Available' : 'Not Available'}</p>
                            {hasPredictions && predictionAge !== null && (
                                <span className={styles.statusDetail}>
                                    {predictionAge === 0 ? 'Today' : `${predictionAge} days ago`}
                                </span>
                            )}
                        </div>
                    </div>
                </div>
            </div>

            {/* Current Sprint Prediction Section */}
            {hasPredictions && (
                <div className={styles.section}>
                    <h2 className={styles.sectionTitle}>Current Sprint Prediction</h2>
                    <div className={styles.predictionGrid} id='overview-sprint-prediction'>
                        {userData.predicted_velocity !== null && (
                            <div className={styles.metricCard}>
                                <div className={styles.metricLabel}>Predicted Velocity</div>
                                <div className={styles.metricValue}>
                                    {userData.predicted_velocity.toFixed(1)}
                                </div>
                                <div className={styles.metricUnit}>story points</div>
                            </div>
                        )}

                        {userData.predicted_sprint_duration_days !== null && (
                            <div className={styles.metricCard}>
                                <div className={styles.metricLabel}>Sprint Duration</div>
                                <div className={styles.metricValue}>
                                    {userData.predicted_sprint_duration_days.toFixed(1)}
                                </div>
                                <div className={styles.metricUnit}>days</div>
                            </div>
                        )}

                        {userData.predicted_duration_days !== null && (
                            <div className={styles.metricCard}>
                                <div className={styles.metricLabel}>Predicted Duration</div>
                                <div className={styles.metricValue}>
                                    {userData.predicted_duration_days.toFixed(1)}
                                </div>
                                <div className={styles.metricUnit}>days</div>
                            </div>
                        )}

                        {userData.confidence_interval !== null && (
                            <div className={styles.metricCard}>
                                <div className={styles.metricLabel}>Confidence</div>
                                <div className={styles.metricValue}>
                                    {(userData.confidence_interval * 100).toFixed(0)}%
                                </div>
                            </div>
                        )}
                    </div>

                    {/* Risk Assessment */}
                    {userData.risk_flag !== null && (
                        <div className={`${styles.riskBanner} ${userData.risk_flag ? styles.riskHigh : styles.riskLow}`}>
                            <span className={styles.riskIcon}>
                                {userData.risk_flag ? '⚠' : '✓'}
                            </span>
                            <div>
                                <strong>{userData.risk_flag ? 'Risk Detected' : 'On Track'}</strong>
                                <p>
                                    {userData.risk_flag
                                        ? 'Sprint may not finish on time based on current trajectory'
                                        : 'Sprint is progressing as expected'}
                                </p>
                            </div>
                        </div>
                    )}

                    {/* LLM Prediction Insight */}
                    {userData.llm_prediction && (
                        <div className={styles.insightCard} id='overview-llm'>
                            <h3 className={styles.insightTitle}>AI Insight</h3>
                            <div
                                className={styles.insightText}
                                dangerouslySetInnerHTML={{
                                    __html: isInsightExpanded || userData.llm_prediction.length <= PREVIEW_LENGTH
                                        ? userData.llm_prediction
                                        : userData.llm_prediction.substring(0, PREVIEW_LENGTH) + '...'
                                }}
                            />
                            {userData.llm_prediction.length > PREVIEW_LENGTH && (
                                <button
                                    className={styles.toggleButton}
                                    onClick={() => setIsInsightExpanded(!isInsightExpanded)}
                                >
                                    {isInsightExpanded ? 'Show Less' : 'Show More'}
                                </button>
                            )}
                        </div>
                    )}
                </div>
            )}

            {/* Account Info Section */}
            <div className={styles.section}>
                <h2 className={styles.sectionTitle}>Account Information</h2>
                <div className={styles.infoGrid}>
                    <div className={styles.infoRow}>
                        <span className={styles.infoLabel}>Provider:</span>
                        <span className={styles.infoValue}>{userData?.provider || 'N/A'}</span>
                    </div>
                    <div className={styles.infoRow}>
                        <span className={styles.infoLabel}>Account ID:</span>
                        <span className={styles.infoValue}>{userData?.account_id || 'N/A'}</span>
                    </div>
                    <div className={styles.infoRow}>
                        <span className={styles.infoLabel}>Member Since:</span>
                        <span className={styles.infoValue}>{formatDate(userData?.created_at)}</span>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default OverviewDisplay;