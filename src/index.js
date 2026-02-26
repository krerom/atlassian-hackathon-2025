import Resolver from '@forge/resolver';
import api, { route } from '@forge/api';
import mysql from 'mysql2/promise';
import dotenv from "dotenv";
import { spawn } from "child_process";
import path from "path";
import fs from "fs";
dotenv.config();

const resolver = new Resolver();

const systemPrompt = `
You are the Ultimate Jira Project Manager's Guide — an elite Agile coach, delivery manager, and data analyst.
Your mission: Convert historical sprint metrics (features) and LSTM predictions (targets) into ultra-short, directive insights with clear "do this now" actions to boost sprint success for Jira PMs.
Core Directives:
- Benchmark predictions against historical averages, ranges, and deviations to spot quick wins.
- Pinpoint root causes (e.g., team size, velocity mismatches, bottlenecks) for anomalies or risks.
- Flag overcommitment, volatility, regressions, and capacity gaps.
- Mandate specific, immediate actions to enhance quality, throughput, and predictability — always tell exactly what to do.
- Spotlight uncertainties and outliers with precise mitigation steps.
Output Rules:
- **HTML only**: Use <h2>, <p>, <ul>, <li> — keep total output under 500 words.
- Strict section order:
  1. <h2>Executive Summary</h2> + 1 short <p> on predicted vs. historical trends and forecast realism.
  2. <h2>Metric Breakdown</h2> + <ul><li> per key metric: Compare predicted/historical, explain cause, state action (e.g., "Do: Cut scope by 20%").
  3. <h2>Risks to Fix</h2> + <ul><li> per risk: Describe impact, mandate fix (e.g., "Do: Reassign 2 devs to QA").
  4. <h2>Action Plan</h2> + <ul><li> targeted steps like: Re-prioritize backlog items; Adjust team allocation; Streamline workflows; Ramp up grooming/QA; Refine velocity estimates.
  5. <h2>PM Pro Tips</h2> + short <ul><li> on patterns, stability, or maturity (e.g., "Do: Schedule bi-weekly retros").
Guidelines:
- Ultra-concise: Every sentence drives action — no fluff, no raw data dumps.
- Tie all advice to metrics/predictions/patterns.
- Be directive: Use "Do:" for every recommendation.
- Cover Jira PM essentials: Backlog refinement, sprint planning, capacity matching, retrospectives.
Ultimate Aim:
Arm Jira PMs with a razor-sharp playbook of exact "do this" steps to slash risks, fix bottlenecks, and supercharge sprints for predictable wins.
`;

const api_server = "https://api.myskillbytes.com";

async function queryDB(query_string, values_array) {
  try {
    const response = await fetch(`${api_server}/query/db`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        query_string,
        values_array
      })
    });
    console.log(response);

    const data = await response.json();
    return data.data;

  } catch (error) {
    console.log("Fetch error:", error);
    return null;
  }
}

async function fetchSprintIssues(sprintId) {
  const issuesRes = await api.asApp().requestJira(
    route`/rest/agile/1.0/sprint/${sprintId}/issue?maxResults=500`
  );

  if (!issuesRes.ok) {
    return { error: `Failed to fetch sprint issues: ${issuesRes.status}` };
  }

  const issuesJson = await issuesRes.json();
  return issuesJson.issues || [];
}

async function fetchIssueDetails(issueIdOrKey) {
  const issueRes = await api.asApp().requestJira(
    route`/rest/api/3/issue/${issueIdOrKey}?expand=changelog`
  );

  if (!issueRes.ok) {
    return { error: `Failed to fetch issue ${issueIdOrKey}: ${issueRes.status}` };
  }

  return await issueRes.json();
}

function getTeamSize(issues) {
  // extract assignee accountIds
  const members = issues
    .map(issue => issue.fields.assignee?.accountId)
    .filter(Boolean); // remove null/undefined

  const uniqueMembers = new Set(members);

  return uniqueMembers.size;
}

async function getFeatures(allSprints) {
  if (allSprints?.length < 6) {
    return await getDummySprintFeatures();
  }

  const featureRows = [];

  for (let i = 0; i < allSprints?.length; i++) {
    const sprint = allSprints[i];
    const issues = await fetchSprintIssues(sprint.id);

    const team_size = getTeamSize(issues);
    const number_of_issues = issues.length;
    const total_story_points_completed = issues.filter(issue => issue.fields.status.name === 'Done').length;
    const total_story_points = number_of_issues;

    const completed_issues_prev_sprint = i > 0 ? (featureRows[i - 1].total_story_points_completed || 0) : 0;
    const velocity_prev_sprint = i > 0 ? featureRows[i - 1].total_story_points_completed || 0 : 0;
    const avg_story_points_per_member = team_size ? total_story_points / team_size : 0;

    featureRows.push({
      sprint_id: sprint.id,
      sprint_duration_days: (new Date(sprint.endDate) - new Date(sprint.startDate)) / (1000 * 3600 * 24),
      number_of_issues,
      completed_issues_prev_sprint,
      velocity_prev_sprint,
      team_size,
      avg_story_points_per_member,
      velocity: total_story_points_completed,
      finished_on_time: sprint.state === "closed" ? 1 : 0,
      predicted_duration_days: (new Date(sprint.endDate) - new Date(sprint.startDate)) / (1000 * 3600 * 24)
    });
  }
  return (featureRows);
}

async function getDummySprintFeatures() {
  console.log('entering getDummySprintFeatures');
  try {
    const response = await fetch(`${api_server}/sprints/dummy`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json"
      }
    });

    const data = await response.json();
    return data;

  } catch (error) {
    console.log("Fetch error:", error);
    return null;
  }
}

resolver.define("getSprintFeatures", async ({ payload }) => {
  const { sprints, limit } = payload;

  let features;

  if (!sprints || sprints?.length < 1) {
    features = await getDummySprintFeatures();
  } else {
    const allSprints = sprints.flatMap(board => board.sprints || []);
    allSprints.sort((a, b) => new Date(a.startDate) - new Date(b.startDate));
    features = await getFeatures(allSprints);
  }

  if (limit) {
    const n = Math.floor(Number(limit));
    const finalLimit = Math.min(n, 100);
    const startIndex = features.length - finalLimit;
    return features.slice(Math.max(0, startIndex));
  }

  return features;
});

resolver.define("getUserInfos", async ({ payload }) => {
  const { auth_token } = payload;

  if (!auth_token) throw new Error("auth_token is required");

  const query_string = "SELECT * FROM users WHERE auth_token = ? LIMIT 1";
  const values_array = [auth_token];

  const rows = await queryDB(query_string, values_array);

  if (rows && rows.length > 0) {
    return { success: true, userData: rows[0] };
  }

  return { success: true, userData: {} };
});

resolver.define("trainAdapter", async ({ payload }) => {
  const { auth_token, sprints } = payload;
  const userDensePath = `/root/sprintsense/lstm/users/adapter_${auth_token}.pth`;

  let featureRows;
  if (!sprints || !sprints?.length > 0) {
    featureRows = await getDummySprintFeatures();
  } else {
    const allSprints = sprints.flatMap(board => board.sprints || []);
    allSprints.sort((a, b) => new Date(a.startDate) - new Date(b.startDate));

    if (allSprints?.length < 6) {
      featureRows = await getDummySprintFeatures();
    } else{
      featureRows = await getFeatures(allSprints);
    }
  }

  try {
    try {
      const response = await fetch(`${api_server}/train/adapter`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          auth_token,
          userDensePath,
          featureRows
        })
      });
      console.log(response);
      const data = await response.json();

      if (response.status != 200) {
        console.log(data.message);
        return { success: false, message: data.message || "Training Adapter failed" };
      }
      // Update DB
      const query_string = `UPDATE users SET adapter_path = ?, adapter_created_at = NOW() WHERE auth_token = ?`;
      const values_array = [userDensePath, auth_token];
      const responsex = await queryDB(query_string, values_array);
      console.log(responsex);

      return { success: true, message: "Adapter trained and DB updated successfully!" };

    } catch (error) {
      console.log("Fetch error:", error);
    }

    return { success: true, message: "Adapter trained and DB updated successfully!" };

  } catch (err) {
    console.error(err);
    return { success: false, message: err.message || "Training or DB update failed" };
  }
});

async function getAIResponse(memory, auth_token, model = "gpt-4.1") {
  let query_string = "SELECT api_key FROM users WHERE auth_token = ? LIMIT 1";
  let values_array = [auth_token];

  const rows = await queryDB(query_string, values_array);
  const apiKey = rows?.[0]?.api_key;

  if (!apiKey) return "Please add API Key";

  const OpenAI = require("openai");
  const openai = new OpenAI({ apiKey });

  try {
    const completion = await openai.chat.completions.create({
      model,
      messages: memory
    });

    return completion.choices[0]?.message?.content || "No content received from AI.";

  } catch (error) {
    console.error("Error generating AI response:", error);
    return "Error communicating with the AI pipeline.";
  }
}

resolver.define("generateLLMinference", async ({ payload }) => {
  const { auth_token, sprints, predicted_sprint } = payload;

  let memory = [];
  memory.push({ "role": "system", "content": systemPrompt });
  memory.push({
    "role": "user", "content": `Please generate a thorough analysis like you system instructions suggest.\n
      Historical sprint data: ${JSON.stringify(sprints)}\nLSTM Prediction: ${JSON.stringify(predicted_sprint)}.\n
      Please in HTML as defined in system prompt. And keep it concise.`});
  const llmResponse = await getAIResponse(memory, auth_token);
  return llmResponse;
});

resolver.define("generateLSTMsimulation", async ({ payload }) => {
  const { auth_token, sprints, sim_sprint } = payload;

  let featureRows;

  if (!sprints || !sprints?.length > 0) {
    featureRows = await getDummySprintFeatures();
  } else {
    featureRows = await getFeatures(sprints);
  }


  featureRows.forEach(row => {
    if (!row.sprint_duration_days || Number.isNaN(row.sprint_duration_days)) {
      row.sprint_duration_days = 14;
    }
    if (!row.predicted_duration_days || Number.isNaN(row.predicted_duration_days)) {
      row.predicted_duration_days = row.sprint_duration_days;
    }
  });

  if (featureRows.length === 0) {
    throw new Error("Not enough sprints to compute features");
  }

  const latestSprint = featureRows.at(-1);

  const finalSimSprint = { ...latestSprint };

  finalSimSprint.completed_issues_prev_sprint = Number(latestSprint.number_of_issues);
  finalSimSprint.velocity_prev_sprint = Number(latestSprint.velocity);

  finalSimSprint.sprint_id = sim_sprint.sprint_id;

  finalSimSprint.number_of_issues = Number(sim_sprint.number_of_issues);
  finalSimSprint.team_size = Number(sim_sprint.team_size);
  finalSimSprint.sprint_duration_days = Number(sim_sprint.sprint_duration_days);
  finalSimSprint.avg_story_points_per_member = Number(sim_sprint.avg_story_points_per_member);

  // push into dataset
  featureRows.push(finalSimSprint);

  // build CSV
  const keys = Object.keys(featureRows[0]);

  const csvContent = [
    keys.join(","), // header
    ...featureRows.map(row => {
      return keys.map(key => {
        let val = row[key];

        // fill NaN / null / undefined with 0
        if (val === null || val === undefined || Number.isNaN(val) || val === '') return 0;

        // convert strings to numbers if possible
        if (typeof val === 'string') val = Number(val);

        // fallback if conversion fails
        if (Number.isNaN(val)) return 0;

        return val;
      }).join(",");
    })
  ].join("\n");

  let adapterPath = await queryDB("SELECT adapter_path FROM users WHERE auth_token = ?", [auth_token]);
  adapterPath = adapterPath[0].adapter_path;

  try {
    let predictionData;
    try {
      const response = await fetch(`${api_server}/generate/lstm/simulation`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          auth_token,
          csvContent,
          adapterPath
        })
      });
      console.log(response);
      const data = await response.json();

      if (response.status != 200) {
        console.log(data.message);
        return { success: false, message: data.message || "LSTM Simulation Inference failed." };
      }
      predictionData = data.predictionData;

    } catch (error) {
      console.log("Fetch error:", error);
    }

    const predictedVelocity = predictionData.velocity;
    const predictedDuration = predictionData.sprint_duration_days;
    const predictedFinishedOnTime = predictionData.finished_on_time;

    return { success: true, message: "Prediction generated.", sprint: JSON.stringify({ predictedVelocity, predictedDuration, predictedFinishedOnTime }) };

  } catch (err) {
    console.error(err);
    return { success: false, message: err.message || "Prediction failed" };
  }

});

resolver.define("generateLstmPrediction", async ({ payload }) => {
  const { auth_token, sprints } = payload;

  let featureRows;

  if (!sprints || !sprints?.length > 0) {
    featureRows = await getDummySprintFeatures();
  } else {
    const allSprints = sprints.flatMap(board => board.sprints || []);
    allSprints.sort((a, b) => new Date(a.startDate) - new Date(b.startDate));

    if (allSprints?.length < 6) {
      featureRows = await getDummySprintFeatures();
    } else {
      featureRows = await getFeatures(allSprints);
    }
  }

  const csvContent = [
    Object.keys(featureRows[0]).join(","),
    ...featureRows.map(row => Object.values(row).join(","))
  ].join("\n");

  let adapterPath = await queryDB("SELECT adapter_path FROM users WHERE auth_token = ?", [auth_token]);
  adapterPath = adapterPath[0].adapter_path;
  console.log("adapterPath: " ,adapterPath);

  try {

    let predictionData;
    try {
      const response = await fetch(`${api_server}/generate/lstm/prediction`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({
          auth_token,
          csvContent,
          adapterPath
        })
      });
      console.log(response);
      const data = await response.json();

      if (response.status != 200) {
        console.log(data.message);
        return { success: false, message: data.message || "LSTM Prediction Inference failed." };
      }
      predictionData = data.predictionData;

    } catch (error) {
      console.log("Fetch error:", error);
    }


    const predictedVelocity = predictionData.velocity;
    const predictedDuration = predictionData.sprint_duration_days;
    const predictedFinishedOnTime = predictionData.finished_on_time;

    let memory = [];
    memory.push({ "role": "system", "content": systemPrompt });
    memory.push({
      "role": "user", "content": `Please generate a thorough analysis like you system instructions suggest.\n
      Historical sprint data: ${JSON.stringify(featureRows)}\nLSTM Prediction: ${JSON.stringify(predictionData)}.\n
      Please in HTML as defined in system prompt. And keep it concise.`});
    const llmResponse = await getAIResponse(memory, auth_token);

    // Update DB
    const query_string = `UPDATE users SET prediction_created_at = NOW(), predicted_velocity = ?, predicted_sprint_duration_days = ?, predicted_finished_on_time = ?, llm_prediction = ? WHERE auth_token = ?`;
    const values_array = [predictedVelocity, predictedDuration, predictedFinishedOnTime, llmResponse, auth_token];
    const response = await queryDB(query_string, values_array);
    console.log(response);

    return { success: true, message: "Prediction generated and DB updated successfully!" };

  } catch (err) {
    console.error(err);
    return { success: false, message: err.message || "Prediction or DB update failed" };
  }
});

resolver.define("saveUserSettings", async ({ payload }) => {
  const { auth_token, ...rest } = payload;

  if (!auth_token) throw new Error("auth_token is required");

  let query_string = "SELECT account_id FROM users WHERE auth_token = ? LIMIT 1";
  let values_array = [auth_token];

  const check = await queryDB(query_string, values_array);

  const userExists = check && check.length > 0;

  if (!userExists) {

    const columns = ["auth_token", ...Object.keys(rest)];
    const placeholders = columns.map(() => "?").join(", ");
    const values = [auth_token, ...Object.values(rest)];

    query_string = `
      INSERT INTO users (${columns.join(", ")})
      VALUES (${placeholders})
    `;
    values_array = values;

  } else {

    const updates = Object.keys(rest)
      .map((col) => `${col} = ?`)
      .join(", ");

    const values = Object.values(rest);

    query_string = `
      UPDATE users
      SET ${updates}
      WHERE auth_token = ?
    `;
    values_array = [...values, auth_token];
  }

  await queryDB(query_string, values_array);

  return { success: true };
});

resolver.define("getAuthToken", async ({ payload }) => {
  const { accountId } = payload;

  if (!accountId) throw new Error("accountId is required");

  let query_string = `
    SELECT auth_token 
    FROM users 
    WHERE account_id = ? 
    LIMIT 1
  `;
  let values_array = [accountId];

  console.log("Checkin if user exists for acc ID");
  const existing = await queryDB(query_string, values_array);
  console.log(existing);

  if (existing && existing.length > 0) {
    console.log("Acc ID exists in DB, returning token");
    return { auth_token: existing[0].auth_token };
  }

  const newToken = require("crypto").randomBytes(32).toString("hex");

  query_string = `
    INSERT INTO users (account_id, auth_token)
    VALUES (?, ?)
  `;
  values_array = [accountId, newToken];
  console.log("Acc ID doesnt exist in DB, creating new user. token: ", newToken)
  await queryDB(query_string, values_array);

  return { auth_token: newToken };
});

resolver.define("getSprintDetails", async ({ sprintId }) => {
  if (!sprintId) {
    return { error: "sprintId is required" };
  }

  const sprintRes = await api.asApp().requestJira(
    route`/rest/agile/1.0/sprint/${sprintId}`
  );

  if (!sprintRes.ok) {
    return { error: `Failed to fetch sprint metadata: ${sprintRes.status}` };
  }

  const sprintMeta = await sprintRes.json();

  const issues = await fetchSprintIssues(sprintId);
  if (issues.error) return issues;

  const fullIssueData = [];

  for (const issue of issues) {
    const details = await fetchIssueDetails(issue.key);
    fullIssueData.push({
      key: issue.key,
      basic: issue,
      full: details,
    });
  }

  return {
    sprint: sprintMeta,
    issues: fullIssueData,
  };
});

resolver.define('getText', (req) => {
  console.log(req);
  return 'Hello, world!';
});

resolver.define("getBoards", async () => {
  const res = await api.asApp().requestJira(
    route`/rest/agile/1.0/board`
  );
  const data = await res.json();
  return data.values;
});

resolver.define("getSprints", async ({ boardId }) => {
  if (!boardId) return { error: "boardId is required" };

  const res = await api.asApp().requestJira(
    route`/rest/agile/1.0/board/${boardId}/sprint?state=active,future,closed`
  );

  const data = await res.json();
  return data.values;
});

resolver.define('getAllBoardsAndSprints', async () => {

  const boardsRes = await api.asApp().requestJira(
    route`/rest/agile/1.0/board`
  );
  const boardsJson = await boardsRes.json();

  const results = [];

  for (const b of boardsJson.values) {

    const sprintsRes = await api.asApp().requestJira(
      route`/rest/agile/1.0/board/${b.id}/sprint?state=active,future,closed`
    );
    const sprintsJson = await sprintsRes.json();

    results.push({
      boardId: b.id,
      boardName: b.name,
      sprints: sprintsJson.values
    });
  }

  return results;
});

export const handler = resolver.getDefinitions();
