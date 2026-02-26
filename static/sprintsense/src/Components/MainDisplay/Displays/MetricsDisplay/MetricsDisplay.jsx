import styles from "./MetricsDisplay.module.css";
import { invoke } from "@forge/bridge";
import React, { useEffect, useState } from "react";
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from "chart.js";
import { Line } from "react-chartjs-2";
import InfoModal from "../../../InfoModal/InfoModal";
import AlertModal from "../../../AlertModal/AlertModal";
import Loader from "../../../Loader/Loader";
import { useTour } from "../../../../utils/TourProvider";

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);


function MetricsDisplay({ sprints, userData, metrics, setMetrics, setUserData, priorMetrics, setPriorMetrics }) {
  const { startTour } = useTour();
  const [loading, setLoading] = useState(true);

  const [limit, setLimit] = useState(10);
  const [triggerReload, setTriggerReload] = useState(0);
  const predictionExists = userData && userData.prediction_created_at != null;
  const [predicting, setPredicting] = useState(false);

  const [showInfo, setshowInfo] = useState(false);
  const [showAlert, setshowAlert] = useState(false);
  const [infoHead, setInfoHead] = useState("");
  const [infoMsg, setInfoMsg] = useState("");
  const [alertHead, setAlertHead] = useState("");
  const [alertMsg, setAlertMsg] = useState("");

  // Chart config
  const labels = metrics.map((m, idx) => `${idx+1}`);
  const options = {
    responsive: true,
    maintainAspectRatio: true,
    scales: {
      x: {
        grid: {
          display: false, // Remove vertical grid lines
          drawBorder: false
        }
      },
      y: {
        grid: {
          color: 'rgba(0, 0, 0, 0.08)', // Faint horizontal lines
        }
      }
    },
    plugins: {
      legend: {
        position: 'top',
        labels: {
          boxWidth: 10,
          usePointStyle: true, // Use dots/lines instead of squares
          font: {
            size: 13
          }
        }
      },
      title: {
        display: false, // Chart title is in the card header
      },
    }
  };

  useEffect(() => {
    setLoading(true);
    async function loadMetrics() {
      const limitExtra = Number(limit) + 6;
      console.log("limitextra: ", limitExtra);
      const features = await invoke("getSprintFeatures", { sprints, limit: limitExtra });
      console.log('Received Metrics:', features);

      const visibleMetrics = features.slice(-limit);
      const priorMetricsx = features.slice(0, -limit);

      setMetrics(visibleMetrics);
      setPriorMetrics(priorMetricsx);

      setLoading(false);
    }
    loadMetrics();
  }, [sprints, triggerReload]);

  useEffect(() => {
    if (metrics?.length > 0) {
      setLoading(false);
      return;
    };
    setLoading(true);
    async function loadMetrics() {
      const limitExtra = Number(limit) + 6;
      console.log("limitextra: ", limitExtra);
      const features = await invoke("getSprintFeatures", { sprints, limit: limitExtra });
      console.log('Received Metrics:', features);
      const visibleMetrics = features.slice(-limit);
      const priorMetricsx = features.slice(0, -limit);

      setMetrics(visibleMetrics);
      setPriorMetrics(priorMetricsx);
      setLoading(false);

      if (!localStorage.getItem("metrics")) {
        startTour(
          [
            {
              target: "#metrics-current",
              content: (
                <>
                  <strong>Current Sprint Performance:</strong><br />
                  View your current sprint metrics, including the <strong>Exponential Moving Average (EMA)</strong>.
                  Click the chart to see a detailed explanation of how to interpret it.
                </>
              ),
            },
            {
              target: "#metrics-limit-select",
              content: (
                <>
                  <strong>Historical Data Range:</strong><br />
                  Adjust the number of past sprints included in your metrics dashboard to analyze trends over different periods.
                </>
              ),
            },
            {
              target: "#metrics-generate",
              content: (
                <>
                  <strong>Generate Predictions:</strong><br />
                  Generate new <strong>LSTM predictions</strong> for your upcoming sprints.
                  This will also provide natural language improvement suggestions.
                </>
              ),
            },
            {
              target: "#metrics-predictions",
              content: (
                <>
                  <strong>Predicted Metrics:</strong><br />
                  The last point in the graph represents the <strong>LSTM prediction</strong>.
                  The first six points show your current sprint data, while the seventh shows the forecast.
                </>
              ),
            },
            {
              target: "#metrics-details",
              content: (
                <>
                  <strong>Detailed Insights:</strong><br />
                  Review a comprehensive, <strong>LLM-generated summary</strong> containing improvement suggestions for your next sprint.
                </>
              ),
            }
          ]

          , "metrics"
        );
      }
    }
    loadMetrics();
  }, []);

  const handleApplyLimit = () => {
    if (limit > 0) {
      setTriggerReload(prev => prev + 1);
    } else {
      setLimit(10);
      setTriggerReload(prev => prev + 1);
    }
  }

  const calculateEMA = (data, period = 6, priorData = []) => {
    if (data.length === 0) return [];

    // Combine prior data with current data for calculation
    const fullData = [...priorData, ...data];

    if (fullData.length < period) {
      return new Array(data.length).fill(0);
    }

    const k = 2 / (period + 1);

    // Calculate initial SMA using first 'period' elements
    let sum = 0;
    for (let i = 0; i < period; i++) {
      sum += fullData[i];
    }
    let ema = sum / period;

    // Calculate EMA for entire dataset starting from index 'period'
    const fullEmaArray = [];

    for (let i = 0; i < fullData.length; i++) {
      if (i < period) {
        fullEmaArray.push(0);
      } else if (i === period) {
        fullEmaArray.push(ema);
      } else {
        ema = fullData[i] * k + ema * (1 - k);
        fullEmaArray.push(Number(ema.toFixed(4)));
      }
    }

    // Return only the EMA values for the visible data range
    // If we have prior data, skip those indices
    const startIndex = priorData.length;
    const result = fullEmaArray.slice(startIndex);

    // Safety check: ensure we return the correct length
    if (result.length !== data.length) {
      console.error('EMA length mismatch', {
        expected: data.length,
        got: result.length,
        priorLength: priorData.length,
        fullLength: fullData.length
      });
    }

    return result;
  };

  const calculateCombinedEMA = (
    array1,
    array2,
    weight1 = 0.5,
    weight2 = 0.5,
    period = 6,
    priorArray1 = [],
    priorArray2 = []
  ) => {
    if (array1.length !== array2.length) {
      throw new Error("Arrays must have the same length");
    }
    if (priorArray1.length !== priorArray2.length) {
      throw new Error("Prior arrays must have the same length");
    }

    // Create combined series for both prior and current data
    const priorCombined = priorArray1.map((val, i) =>
      val * weight1 + priorArray2[i] * weight2
    );
    const combined = array1.map((val, i) =>
      val * weight1 + array2[i] * weight2
    );

    return calculateEMA(combined, period, priorCombined);
  };

  const handleStartPrediction = async () => {
    setPredicting(true)
    setLoading(true);
    try {
      console.log(`AUthToken: ${userData.auth_token}`)
      const response = await invoke("generateLstmPrediction", { auth_token: userData.auth_token, sprints });
      if (response.success == false) {
        setAlertHead("Could not generate prediction");
        setAlertMsg(response.message);
        setshowAlert(true);
      }

      const userInfoResponse = await invoke("getUserInfos", { auth_token: userData.auth_token });
      if (!userInfoResponse || userInfoResponse.success === false) {
        console.warn("Could not retrieve user data.");
      }
      setUserData(userInfoResponse.userData || {});

      console.log(JSON.stringify(response));
    } catch (error) {
      console.error('An Error occured: ', error);
      setAlertHead("Please reload");
      setAlertMsg("Generating your analysis took a little longer than expected. Please reload in a couple seconds to get latest data from the server.")
      setshowAlert(true);
      return null;
    }

    finally {
      setPredicting(false);
      setLoading(false);
    }
  };

  const handleDaysTakenInfo = () => {
    setInfoHead("Days Taken per Sprint + EMA-6 – Delivery Speed Trend");
    function createChartMockupArray() {
      const charts = [];

      const sprintIDs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
      const daysTaken = [4, 5, 6, 5, 4, 6, 5, 7, 6, 5];
      const emaRising = [1, 2, 3, 4, 6, 7, 8, 9, 8, 7];
      const emaFlat = [4.5, 4, 5, 5.5, 4.5, 4, 5, 4, 5, 6];
      const emaFalling = [7, 6, 4, 5, 5, 4, 4, 3, 2, 2];

      const SPRINT_COLOR = "rgba(26, 115, 232, 0.8)";
      const EMA_COLOR = "rgba(234, 67, 53, 0.8)";

      const data1 = {
        labels: sprintIDs,
        datasets: [
          {
            label: "Sprint Duration (days)",
            data: daysTaken,
            borderColor: SPRINT_COLOR,
            backgroundColor: SPRINT_COLOR,
          },
          {
            label: "Exponential Moving Average (EMA)",
            data: emaRising,
            borderColor: EMA_COLOR,
            backgroundColor: EMA_COLOR,
          }
        ],
      };
      const data2 = {
        labels: sprintIDs,
        datasets: [
          {
            label: "Sprint Duration (days)",
            data: daysTaken,
            borderColor: SPRINT_COLOR,
            backgroundColor: SPRINT_COLOR,
          },
          {
            label: "Exponential Moving Average (EMA)",
            data: emaFlat,
            borderColor: EMA_COLOR,
            backgroundColor: EMA_COLOR,
          }
        ],
      };
      const data3 = {
        labels: sprintIDs,
        datasets: [
          {
            label: "Sprint Duration (days)",
            data: daysTaken,
            borderColor: SPRINT_COLOR,
            backgroundColor: SPRINT_COLOR,
          },
          {
            label: "Exponential Moving Average (EMA)",
            data: emaFalling,
            borderColor: EMA_COLOR,
            backgroundColor: EMA_COLOR,
          }
        ],
      };
      const graph1 = () => { return (<Line data={data1} options={options} height={50} />) };
      const graph2 = () => { return (<Line data={data2} options={options} height={50} />) };
      const graph3 = () => { return (<Line data={data3} options={options} height={50} />) };
      charts.push(graph1());
      charts.push(graph2());
      charts.push(graph3());
      return charts;
    }
    const createMessage = () => {
      return (
        <>
          <p>This chart shows <strong>how many calendar days</strong> each sprint actually lasted <span style={{ color: "rgb(29, 78, 216)" }}>(blue line)</span> and its <strong>6-sprint</strong> Exponential Moving Average <span style={{ color: "rgb(220, 38, 38)" }}>(red line)</span>.</p>
          <h2>How to read the EMA</h2>
          <ul>
            <li>
              <strong>Rising EMA:&nbsp;</strong>
              Sprints are taking longer over time. The team is slowing down. Typical reasons: growing complexity, more carry-over, blockers, holidays, or over-commitment.
            </li>
            <li>
              <strong>EMA above your target sprint length:&nbsp;</strong>
              Sprints consistently run long — capacity or commitment needs attention.
            </li>
            {createChartMockupArray()[0]}
            <li>
              <strong>Stable / Flat EMA:&nbsp;</strong>
              Ideal state. Sprint duration is predictable — planning and forecasting become reliable.
            </li>
            {createChartMockupArray()[1]}
            <li>
              <strong>Falling EMA:&nbsp;</strong>
              Positive trend. The team is closing sprints faster. Usually means better focus, fewer blockers, reduced carry-over, or improved estimation/flow.
            </li>
            <li>
              <strong>EMA below your target sprint length (e.g. below 10 or 14 days):&nbsp;</strong>
              Sprints regularly finish early — very healthy signal.
            </li>
            {createChartMockupArray()[2]}
          </ul>
          <h2>Key takeaway</h2>
          <p>
            You want the orange EMA line as low and stable (or gently falling) as possible. A steadily
            rising EMA is the clearest
            early warning that delivery speed
            is degrading, even if velocity still looks okay.
            Use this trend in retrospectives and when discussing sustainable
            pace with the team.
          </p>
        </>
      );
    };
    setInfoMsg(createMessage());

    setshowInfo(true);
  };

  const handleVelocityVsIssuesInfo = () => {
    setInfoHead("Velocity and Issues + EMA-6 – Delivery Speed Trend");
    function createChartMockupArray() {
      const charts = [];

      const sprintIDs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
      const velocity = [4, 5, 6, 5, 4, 6, 5, 7, 6, 5];
      const issues = [5, 6, 5, 4, 5, 6, 7, 6, 4, 6];
      const emaRising = [1, 2, 3, 4, 6, 7, 8, 9, 8, 7];
      const emaFlat = [4.5, 4, 5, 5.5, 4.5, 4, 5, 4, 5, 6];
      const emaFalling = [7, 6, 4, 5, 5, 4, 4, 3, 2, 2];

      const issue_color = "rgba(255, 159, 28, 0.4)";
      const velocity_color = "rgba(26, 115, 232, 0.8)";
      const EMA_COLOR = "rgba(234, 67, 53, 0.8)";


      const data1 = {
        labels: sprintIDs,
        datasets: [
          {
            label: "Velocity",
            data: velocity,
            borderColor: velocity_color,
            backgroundColor: velocity_color,
          },
          {
            label: "Issues",
            data: issues,
            borderColor: issue_color,
            backgroundColor: issue_color,
          },
          {
            label: "Exponential Moving Average (EMA)",
            data: emaRising,
            borderColor: EMA_COLOR,
            backgroundColor: EMA_COLOR,
          }
        ],
      };
      const data2 = {
        labels: sprintIDs,
        datasets: [
          {
            label: "Velocity",
            data: velocity,
            borderColor: velocity_color,
            backgroundColor: velocity_color,
          },
          {
            label: "Issues",
            data: issues,
            borderColor: issue_color,
            backgroundColor: issue_color,
          },
          {
            label: "Exponential Moving Average (EMA)",
            data: emaFlat,
            borderColor: EMA_COLOR,
            backgroundColor: EMA_COLOR,
          }
        ],
      };
      const data3 = {
        labels: sprintIDs,
        datasets: [
          {
            label: "Velocity",
            data: velocity,
            borderColor: velocity_color,
            backgroundColor: velocity_color,
          },
          {
            label: "Issues",
            data: issues,
            borderColor: issue_color,
            backgroundColor: issue_color,
          },
          {
            label: "Exponential Moving Average (EMA)",
            data: emaFalling,
            borderColor: EMA_COLOR,
            backgroundColor: EMA_COLOR,
          }
        ],
      };
      const graph1 = () => { return (<Line data={data1} options={options} height={50} />) };
      const graph2 = () => { return (<Line data={data2} options={options} height={50} />) };
      const graph3 = () => { return (<Line data={data3} options={options} height={50} />) };
      charts.push(graph1());
      charts.push(graph2());
      charts.push(graph3());
      return charts;
    }
    const createMessage = () => {
      return (
        <>
          <div className="chart-intro">
            <p>
              This chart visualizes each sprint’s
              <strong> Velocity </strong>
              <span style={{ color: "rgb(29, 78, 216)" }}>(blue line)</span>,
              <strong> Completed Issues </strong>
              <span style={{ color: "rgb(255, 159, 28)" }}>(orange line)</span>,
              and a combined
              <strong> 50/50 Exponential Moving Average </strong>
              <span style={{ color: "rgb(220, 38, 38)" }}>(red line)</span>.
            </p>

            <p className="subtle">
              The blended EMA smooths out noisy sprint-to-sprint variation and reveals your team’s
              underlying delivery trend more clearly than raw velocity alone.
            </p>
          </div>

          <h2>How to read the EMA</h2>

          <ul className="insight-list">
            <li>
              <strong>Rising EMA:&nbsp;</strong>
              Delivery output (velocity + completed issues) is improving.
              Often linked to fewer blockers, better flow, or improved estimation stability.
            </li>

            <li>
              <strong>EMA above recent baseline:&nbsp;</strong>
              The team is consistently performing above earlier sprints — a strong sign of
              maturing processes and stable capacity.
            </li>

            {createChartMockupArray()[0]}

            <li>
              <strong>Flat / Stable EMA:&nbsp;</strong>
              Predictable and steady delivery.
              Ideal for planning because sprint capacity becomes highly reliable.
            </li>

            {createChartMockupArray()[1]}

            <li>
              <strong>Falling EMA:&nbsp;</strong>
              A decline in combined output, often caused by interruptions, scope churn,
              increased complexity, or reduced availability.
            </li>

            <li>
              <strong>EMA below expected range:&nbsp;</strong>
              Sustained low output is an early signal to inspect capacity, blockers,
              or misaligned sprint scope.
            </li>

            {createChartMockupArray()[2]}
          </ul>

          <h2>Key takeaway</h2>
          <p>
            The blended EMA gives a clearer, more stable picture of your team’s
            <strong> true delivery trajectory </strong>.
            A rising EMA indicates growing predictability and throughput, while a declining EMA
            often exposes delivery problems *before* velocity drops become obvious.
          </p>
        </>
      );
    };
    setInfoMsg(createMessage());

    setshowInfo(true);
  };

  const renderCurrentPerformaceCharts = () => {
    const VELOCITY_COLOR = "rgba(26, 115, 232, 0.8)";
    const ISSUE_COLOR = "rgba(255, 159, 28, 0.4)";
    const SPRINT_COLOR = "rgba(26, 115, 232, 0.8)";
    const EMA_COLOR = "rgba(234, 67, 53, 0.8)";

    // Extract prior data for EMA calculation
    const priorSprintDuration = priorMetrics.map(m => Number(m.sprint_duration_days ?? 0));
    const priorVelocity = priorMetrics.map(m => Number(m.velocity ?? 0));
    const priorIssues = priorMetrics.map(m => Number(m.number_of_issues ?? 0));

    const dataDaysTakenPerSprint = {
      labels,
      datasets: [
        {
          label: "Sprint Duration (days)",
          data: metrics.map((m) => Number(m.sprint_duration_days ?? 0)),
          borderColor: SPRINT_COLOR,
          backgroundColor: SPRINT_COLOR,
        },
        {
          label: "Exponential Moving Average (EMA)",
          data: calculateEMA(
            metrics.map((m) => Number(m.sprint_duration_days ?? 0)),
            6,
            priorSprintDuration
          ),
          borderColor: EMA_COLOR,
          backgroundColor: EMA_COLOR,
        }
      ],
    };

    const dataVelocityIssues = {
      labels,
      datasets: [
        {
          label: "Velocity",
          data: metrics.map((m) => Number(m.velocity ?? 0)),
          borderColor: VELOCITY_COLOR,
          backgroundColor: VELOCITY_COLOR,
        },
        {
          label: "Number of Issues",
          data: metrics.map((m) => Number(m.number_of_issues ?? 0)),
          borderColor: ISSUE_COLOR,
          backgroundColor: ISSUE_COLOR,
        },

        {
          label: "Exponential Moving Average (EMA)",
          data: calculateCombinedEMA(
            metrics.map((m) => Number(m.velocity ?? 0)),
            metrics.map((m) => Number(m.number_of_issues ?? 0)),
            0.5,
            0.5,
            6,
            priorVelocity,
            priorIssues
          ),
          borderColor: EMA_COLOR,
          backgroundColor: EMA_COLOR,
        },
      ],
    };

    return (
      <>
        <div className={styles.chartCard}>
          <h3 className={styles.chartTitle} onClick={handleDaysTakenInfo}>Days Taken per Sprint ℹ️</h3>
          <div className={styles.chartWrapper}>
            <Line data={dataDaysTakenPerSprint} options={options} />
          </div>
          <p className={styles.infos}>The EMA needs 6 prior values to calculate. This is why the first 6 EMA values for a dataset are undefined.</p>
        </div>

        <div className={styles.chartCard} >
          <h3 className={styles.chartTitle} onClick={handleVelocityVsIssuesInfo}>Velocity (Issues Completed) ℹ️</h3>
          <div className={styles.chartWrapper}>
            <Line data={dataVelocityIssues} options={options} />
          </div>
        </div>
      </>
    );
  }

  const renderPredictionChart = () => {
    const historicalMetrics = metrics.slice(-6);

    const HISTORICAL_COLOR = 'rgb(29, 78, 216)';
    const PREDICTION_COLOR = 'rgb(139, 92, 246)';
    const SUCCESS_COLOR = 'rgb(34, 153, 84)';
    const WARNING_COLOR = 'rgb(255, 159, 28)';
    const FAILURE_COLOR = 'rgba(234, 67, 53, 0.8)';
    const LAST_POINT_COLOR = 'rgb(251, 191, 36)';

    const historicalLabels = historicalMetrics.map((m, index) => `Sprint ${metrics.length - 5 + index}`);
    const labels = [...historicalLabels, `Sprint ${historicalMetrics.length + 1} (Predicted)`];

    const historicalVelocity = historicalMetrics.map(m => Number(m.velocity ?? 0));
    const predictedVelocity = Number(userData.predicted_velocity ?? 0);
    const fullVelocityData = [...historicalVelocity, predictedVelocity];

    const historicalSprintDuration = historicalMetrics.map(m => Number(m.sprint_duration_days ?? 0));
    const predictedSprintDuration = Number(userData.predicted_sprint_duration_days ?? 0);
    const fullSprintDurationData = [...historicalSprintDuration, predictedSprintDuration];

    const historicalFOT = historicalMetrics.map(m => m.finished_on_time ? 1 : 0);
    const predictedFOT = Number(userData.predicted_finished_on_time > 1 ? 1 : 0); // Assuming 1 or 0 is saved to DB
    const fullFOTData = [...historicalFOT, predictedFOT];

    const totalLength = fullFOTData.length;

    const pointBackgroundColors = fullFOTData.map(val =>
      val === 1 ? HISTORICAL_COLOR : FAILURE_COLOR
    );

    const pointBorderColors = [...pointBackgroundColors]; // nice visible yellow

    pointBackgroundColors[totalLength - 1] = PREDICTION_COLOR;
    pointBorderColors[totalLength - 1] = PREDICTION_COLOR;

    const pointRadius = fullFOTData.map((_, i) => i === totalLength - 1 ? 8 : 4);
    const pointBorderWidth = fullFOTData.map((_, i) => i === totalLength - 1 ? 3 : 1);

    const borderColors = Array(historicalVelocity.length).fill(HISTORICAL_COLOR);
    borderColors.push(PREDICTION_COLOR); // Set the last point to Red

    const backgroundColors = Array(historicalVelocity.length).fill(HISTORICAL_COLOR);
    backgroundColors.push(PREDICTION_COLOR); // Set the last point to Red

    // 5. Create the FINAL Velocity Data Structure
    const velocityDataset = {
      label: "Velocity (Historical & Predicted)",
      data: fullVelocityData,
      // Crucial: Use the color arrays to style each point individually
      borderColor: borderColors,
      backgroundColor: backgroundColors.map(c => c.replace('rgb', 'rgba').replace(')', ', 0.5)')), // Use semi-transparent background
      borderWidth: 2,
      pointBackgroundColor: borderColors,
      pointBorderColor: borderColors,
      pointRadius: 5, // Make points visible
      pointHoverRadius: 7,
      tension: 0.1
    };
    const sprintDurationDataset = {
      label: "Duration days (Historical & Predicted)",
      data: fullSprintDurationData,
      // Crucial: Use the color arrays to style each point individually
      borderColor: borderColors,
      backgroundColor: backgroundColors.map(c => c.replace('rgb', 'rgba').replace(')', ', 0.5)')), // Use semi-transparent background
      borderWidth: 2,
      pointBackgroundColor: borderColors,
      pointBorderColor: borderColors,
      pointRadius: 5, // Make points visible
      pointHoverRadius: 7,
      tension: 0.1
    };
    const finishedOnTimeDataset = {
      label: "Finished On Time (1=Yes, 0=No)",
      data: fullFOTData,
      borderColor: "rgba(75, 192, 192, 0)", // Neutral line color
      stepped: true, // Use step chart style

      // Dynamic point styling
      pointRadius: 6,
      pointHoverRadius: 8,
      pointBackgroundColor: pointBackgroundColors,
      pointBorderColor: pointBackgroundColors,
      pointBorderWidth: 2,
    };

    // 6. Create the final data object for the chart
    const dataVelocity = {
      labels: labels,
      datasets: [
        velocityDataset
      ],
    };
    const dataSprintsDur = {
      labels: labels,
      datasets: [
        sprintDurationDataset
      ],
    };
    const dataFinishedOnTime = {
      labels: labels,
      datasets: [finishedOnTimeDataset],
    };

    // Assuming you have access to 'options'
    return (
      <>
        <div className={styles.chartCard}>
          <h3 className={styles.chartTitle}>Predicted Velocity</h3>
          <div className={styles.chartContainer}>
            <Line data={dataVelocity} options={options} />
          </div>
        </div>

        <div className={styles.chartCard}>
          <h3 className={styles.chartTitle}>Predicted Sprint Duration</h3>
          <div className={styles.chartContainer}>
            <Line data={dataSprintsDur} options={options} />
          </div>
        </div>

        <div className={styles.chartCard}>
          <h3 className={styles.chartTitle}>On-Time Completion Probability</h3>
          <div className={styles.chartContainer}>
            <Line data={dataFinishedOnTime} options={options} />
          </div>
        </div>
      </>
    );
  };

  const showDetails = () => {
    setInfoHead("Detailed Analysis");
    setInfoMsg(<span dangerouslySetInnerHTML={{ __html: userData.llm_prediction }}></span>);
    setshowInfo(true);
  };

  if (loading) return (<div className={styles.dashboard}><Loader /></div>);
  if (!Array.isArray(metrics) || metrics.length === 0) {
    return "No metrics data available to display.";
  }
  return (
    <div className={styles.dashboard}>
      {showInfo && <InfoModal setshowInfo={setshowInfo} head={infoHead} message={infoMsg} />}
      {showAlert && <AlertModal setshowAlert={setshowAlert} head={alertHead} message={alertMsg} />}
      {/* Header with Limit Selector */}
      <div className={styles.header}>
        <h1 className={styles.title}>Sprint Analytics Dashboard</h1>
        <div className={styles.limitSelector} id="metrics-limit-select">
          <span className={styles.label}>Sprints to display:</span>
          {[10, 20, 50, 100].map((val) => (
            <button
              key={val}
              className={`${styles.limitBtn} ${limit === val ? styles.limitBtnActive : ''}`}
              onClick={() => setLimit(Number(val))}
            >
              {val}
            </button>
          ))}
          <input
            type="number"
            min="1"
            max="100"
            placeholder="Custom"
            value={limit === "ALL" ? "" : limit}
            onChange={(e) => {
              const val = e.target.value;
              if (val === "") setLimit("ALL");
              else if (Number(val) <= 100) setLimit(Number(val));
            }}
            className={styles.customInput}
          />
          <button onClick={handleApplyLimit} className={styles.applyBtn}>
            Apply
          </button>
          <span className={styles.note}>Max 100 sprints</span>
          <button title="Reload" onClick={() => setTriggerReload(prev => prev + 1)} >🔄</button>
        </div>
      </div>

      {/* Current Metrics Section */}
      <section className={styles.section} id="metrics-current">
        <h2 className={styles.sectionTitle}>Current Performance</h2>
        <div className={styles.grid2}>
          {renderCurrentPerformaceCharts()}
        </div>
      </section>

      {/* Prediction Section */}
      <section className={styles.section} id="metrics-predictions">
        <div className={styles.sectionHeader}>
          <h2 className={styles.sectionTitle}>Predictions & Forecasting</h2>
          <button onClick={showDetails} className={styles.predictBtn} id="metrics-details">
            Details
          </button>
          <button onClick={handleStartPrediction} className={styles.predictBtn} id="metrics-generate">
            Generate New Prediction
          </button>
        </div>

        {predictionExists && (
          <div className={styles.grid3}>
            {renderPredictionChart()}
          </div>
        )}
      </section>
    </div>
  );
}

export default MetricsDisplay;