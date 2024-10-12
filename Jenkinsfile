pipeline {
    agent any

    environment {
        PYTHON = 'python3.11.7'  
    }

    stages {
        stage('Setup Environment') {
            steps {
                sh '''
                    python -m venv venv  # Create a virtual environment
                    . venv/bin/activate   # Activate the virtual environment
                    pip install --upgrade pip
                    pip install -r requirements.txt  # Install dependencies
                '''
            }
        }

        stage('Run Training Script') {
            steps {
                sh '''
                    . venv/bin/activate
                    python train.py  # Run the training script
                '''
            }
        }

        stage('Archive Results') {
            steps {
                archiveArtifacts artifacts: 'Results/*', allowEmptyArchive: true  // Archive metrics and plots
                archiveArtifacts artifacts: 'Model/*', allowEmptyArchive: true    // Archive the model
            }
        }
    }

    post {
        always {
            // Clean up virtual environment
            deleteDir()
        }
    }
}
