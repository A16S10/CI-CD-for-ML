pipeline {
    agent any

    environment {
        PYTHON = '/usr/local/bin/python3.11'  
    }
    
    stages {
        stage('Setup Environment') {
            steps {
                sh '''
                    ${PYTHON} -m venv venv  # Create a virtual environment
                    . venv/bin/activate
                    pip install --upgrade pip
                    pip install -r requirements.txt  # Install dependencies
                '''
            }
        }

        stage('Load Datasets') {
            steps {
                script {
                    def datasets = [
                        "/var/lib/jenkins/workspace/CI-CD-for-ML/Data/drug200.csv", 
                        "/var/lib/jenkins/workspace/CI-CD-for-ML/Data/dataset201.csv", 
                        "/var/lib/jenkins/workspace/CI-CD-for-ML/Data/dataset202.csv"
                    ] 
                    
                    for (dataset in datasets) {
                        echo "Processing ${dataset}"
                        sh """
                            . venv/bin/activate 
                            ${PYTHON} train.py "${dataset}"  # Call your training script with the dataset path as an argument
                        """
                    }
                }
            }
        }
        
        stage('Archive Results') {
            steps {
                archiveArtifacts artifacts: 'Results/**/metrics.txt', fingerprint: true  // Archive all metrics files generated during training
                archiveArtifacts artifacts: 'Model/**/*.pkl', fingerprint: true  // Archive all saved models
            }
        }
        
        stage('Notify') {
            steps {
                echo 'Training and evaluation completed for all datasets.'
            }
        }
    }
}
