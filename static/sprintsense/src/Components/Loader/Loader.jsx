import styles from "./Loader.module.css"

function Loader(){
    return(
        <div className={styles.rippleLoader}>
            <div className={styles.dot}></div>
            <div className={styles.dot}></div>
            <div className={styles.dot}></div>
        </div>
    );
}

export default Loader;