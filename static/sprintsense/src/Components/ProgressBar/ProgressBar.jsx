const ProgressBar = ({ value, max }) => {
    // Prevent divide-by-zero
    const percent = max > 0 ? Math.min((value / max) * 100, 100) : 0;
  
    return (
      <div style={{
        width: "100%",
        height: "12px",
        background: "#e5e7eb",
        borderRadius: "6px",
        overflow: "hidden"
      }}>
        <div style={{
          width: `${percent}%`,
          height: "100%",
          background: "#3b82f6",
          transition: "width 0.3s ease"
        }} />
      </div>
    );
  };
export default ProgressBar;