# Streamlit App for Data Visualization and Machine Learning

## Features
1. Upload CSV File (Done)
2. Data Visualization (Ongoing)
3. Data Cleanup
4. Feature Selection (Basic Version Done)
5. Classification (Basic Version Done)

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


## Related Resources

- [**awesome-streamlit**](https://github.com/MarcSkovMadsen/awesome-streamlit): List of Awesome Streamlit Apps
- [**Streamlit App Gallery**](https://www.streamlit.io/gallery): Official Streamlit Apps Gallery
- [**Best-of**](https://best-of.org): Best-of lists with Python and other languages
- [**Best-of Streamlit**](https://github.com/jrieke/best-of-streamlit): Best-of list of Streamlit Apps

```
  ____  _                 _          ____             
 / ___|| |__   __ _ _ __ | |_ ___   |  _ \ ___  _   _ 
 \___ \| '_ \ / _` | '_ \| __/ _ \  | |_) / _ \| | | |
  ___) | | | | (_| | | | | || (_) | |  _ < (_) | |_| |
 |____/|_| |_|\__,_|_| |_|\__\___/  |_| \_\___/ \__, |
                                                |___/ 
```

Created TextArt using [patorjk.com](https://patorjk.com/software/taag/#p=display&f=Graffiti&t=Type%20Something%20)