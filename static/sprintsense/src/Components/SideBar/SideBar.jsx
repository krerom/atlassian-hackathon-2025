import React, { useEffect } from 'react';
import styles from './SideBar.module.css';
import { motion, useAnimation } from 'framer-motion';
import { useTour } from '../../utils/TourProvider';

function SideBar({ setPage, page, settourResKey }) {
    const { startTour } = useTour();
    const controls = useAnimation();

    const menuItems = [
        { id: 'overview', label: 'Overview' },
        { id: 'metrics', label: 'Metrics' },
        { id: 'simulation', label: 'Simulation' },
        { id: 'settings', label: 'Settings' },
        { id: 'faq', label: 'FAQs' },
    ];

    const listVariants = {
        hidden: {},
        show: {
            transition: {
                staggerChildren: 0.06, // fast stagger (Apple-style)
                delayChildren: 0.05
            }
        }
    };

    const itemVariants = {
        hidden: { opacity: 0, y: 6, filter: "blur(2px)" },
        show: {
            opacity: 1,
            y: 0,
            filter: "blur(0px)",
            transition: {
                type: "spring",
                stiffness: 110,
                damping: 20,
                mass: 0.5
            }
        }
    };


    useEffect(() => {
        controls.start("show");
        const timeout = setTimeout(() => {
            if (!localStorage.getItem("menu")) {
                startTour(
                    [
                        {
                            target: "#overview",
                            content: (
                                <>
                                    <strong>Overview:</strong><br />
                                    This section provides a summary of your user account,
                                    giving you a quick glance at your current sprints and key information.
                                </>
                            ),
                        },
                        {
                            target: "#metrics",
                            content: (
                                <>
                                    <strong>Metrics:</strong><br />
                                    Here you can view detailed metrics calculated from your sprint performance,
                                    helping you track progress and identify trends over time.
                                </>
                            ),
                        },
                        {
                            target: "#simulation",
                            content: (
                                <>
                                    <strong>Simulation:</strong><br />
                                    Experiment with your next sprint by adjusting parameters and seeing predicted outcomes.
                                    This lets you explore different scenarios safely.
                                </>
                            ),
                        },
                        {
                            target: "#settings",
                            content: (
                                <>
                                    <strong>Settings:</strong><br />
                                    Add your <strong>OpenAI API Key</strong> and train your LSTM Adapter
                                    to make accurate predictions for upcoming sprints.
                                </>
                            ),
                        },
                        {
                            target: "#faq",
                            content: (
                                <>
                                    <strong>FAQs:</strong><br />
                                    Find answers to common questions about the app, tours, and sprint simulations here.
                                </>
                            ),
                        }
                    ]
                    ,
                    "menu");
            }
        }, 400);

        return () => clearTimeout(timeout);
    }, []);

    return (
        <nav className={styles.container} aria-label="Main navigation">
            <div className={styles.brand}>
                <div className={styles.logoIcon}>S</div>
                <h1 className={styles.brandName}>SprintSense</h1>
            </div>

            <motion.div
                className={styles.menuList}
                variants={listVariants}
                initial="hidden"
                animate={controls}
                id='menuList'
            >
                {menuItems.map((item) => (
                    <motion.button
                        key={item.id}
                        id={item.id}
                        variants={itemVariants}
                        className={`${styles.menuItem} ${page === item.id ? styles.active : ''}`}
                        onClick={() => setPage(item.id)}
                    >
                        {item.label}
                    </motion.button>
                ))}
            </motion.div>
        </nav>
    );
}

export default SideBar;