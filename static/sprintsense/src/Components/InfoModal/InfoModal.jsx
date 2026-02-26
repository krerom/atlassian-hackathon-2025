import styles from "./InfoModal.module.css"
import React, {useState} from "react"

function InfoModal({head, message, setshowInfo}){
    return(
        <div className={styles.container}>
            <div className={styles.modal}>
                <div className={styles.head}>
                    <span>{head}</span>
                    <button className={styles.closeBtn} onClick={()=>setshowInfo(false)}>✖️</button>
                </div>
                <div className={styles.message}>
                    {message}
                </div>
            </div>
        </div>
    );
}

export default InfoModal;