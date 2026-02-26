import styles from "./AlertModal.module.css"

function AlertModal({head, message, setshowAlert}){
    return(
        <div className={styles.container}>
            <div className={styles.modal}>
                <div className={styles.head}>
                    <span>{head}</span>
                    <button className={styles.closeBtn} onClick={()=>setshowAlert(false)}>✖️</button>
                </div>
                <div className={styles.message}>
                    <span>{message}</span>
                </div>
            </div>
        </div>
    );
}

export default AlertModal;