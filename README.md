# NEMO_II
NEMO_II Repository

# About NEMO_II
The Never-Ending Medical Operative (NEMO) is a system under development to implement Lifelong Machine Learning in the healthcare domain. <br/>

# Version History: <br/>
Version 1.0 (Current) - Currently performs Automated Machine Learning to optimize a Artificial Neural Network.

Pull the dev branch for the development version of NEMO. Use at your own risk! 

# Setup (only done once)
1) Install the MySQL database server on the machine running NEMO.<br/>
2) Recommended: Create a NEMO specific account. Remember the credentials. <br/>
   Not Recommended (but possible): Use the root MySQL account. 
3) Within MySQL, run KBCreator.sql <br/>
   This creates the AlgorithmResults table. The DATA table will be created when NEMO is executed. <br/>
   

# How to Run <br/>
1) Visit the README.md files in the data and config directories and follow the appropriate instructructions.
2) Once the data files and configuration files are set up, pull from GitHub to ensure you have the most recent version.
3) Run: python NEMO.py

To quit: Press Ctrl+C or equivalent. Currently, it takes  few tries to shut off the system.

# TO-DO
1) Implement better exit, kill, and restart behavior.
2) Implement a menu 
3) Investigate the coordinate ascent algorithm and fix any bugs.
4) Extend the documentation
5) Add a new algorithm
6) Complete any TO-DOs in the data and/or config sections

