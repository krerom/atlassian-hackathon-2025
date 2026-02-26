import styles from "./SprintsDisplay.module.css"

function SprintsDisplay({sprints}){

    const onSelectSprint = (sprintID) => {
        console.log(sprintID);
    };

    return (
        <div className={styles.container}>
            {sprints?.map(board => (
                <div key={board.boardId} className={styles.boardSection}>
                    <h2 className={styles.boardName}>{board.boardName}</h2>

                    <div className={styles.cardsSection}>
                        {board.sprints?.map(sprint => (
                            <div
                                key={sprint.id}
                                className={styles.card}
                                onClick={() => onSelectSprint(sprint.id)}
                            >
                                <h3>{sprint.name}</h3>
                                <p>ID: {sprint.id}</p>
                                <p>Status: {sprint.state}</p>
                            </div>
                        ))}
                    </div>
                </div>
            ))}
        </div>
    );
}

export default SprintsDisplay;