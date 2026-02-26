import { useEffect, useState } from "react";

export default function useAuthToken(jiraEmail) {
  const [authToken, setAuthToken] = useState(null);

  useEffect(() => {
    const localToken = localStorage.getItem("auth_token");
    if (localToken) {
      setAuthToken(localToken);
      return; // token exists, no need to fetch
    }

    const fetchAuthToken = async () => {
      try {
        const response = await fetch("/api/get_auth_token", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ email: jiraEmail }),
        });

        const data = await response.json();

        if (data.auth_token) {
          localStorage.setItem("auth_token", data.auth_token);
          setAuthToken(data.auth_token);
        } else {
          console.warn("No auth token found for this user");
        }
      } catch (err) {
        console.error("Error fetching auth token:", err);
      }
    };

    if (jiraEmail) {
      fetchAuthToken();
    }
  }, [jiraEmail]);

  return authToken;
}
