# Streamlit App for Data Visualization and Machine Learning

## Features
1. Upload CSV File (Done)
2. Data Visualization (Ongoing)
3. Data Cleanup
4. Feature Selection
5. Classification

## Initial Set Up
### Using Docker
* Build the Container using Docker

    ```
    $ docker image build -t streamlit:app .
    ```

* Now, run the docker image on port 8501 (default for Streamlit Apps)
    ```
    $ docker container run -p 8501:8501 -d streamlit:app
    ```

### Without Docker
* Create and Activate Virtual Environment using Conda
    ```
    $ conda create --name streamlit_app python=3.7
    $ source activate streamlit_app
    ```
    
    or venv
    ```
    $ python3 -m venv streamlit_app
    $ source streamlit_app/bin/activate
    ```

* Install the `requirements.txt` file
    ```
    $ pip install -r requirements.txt
    ```

* Run the app
    ```
    $ streamlit run app.py
    ```