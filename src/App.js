import _ from "lodash";
import React, { Component } from "react";
import { buildStyles, CircularProgressbar } from "react-circular-progressbar";
import { toast } from "react-toastify";
import voca from "voca";
import modelWorking from "./assets/images/ModelWorking.png";
import trainingDataset from "./assets/images/TrainingDataset.png";
import axiosInstance from "./axios";

// function getRandomColor() {
//   var letters = "0123456789ABCDEF";
//   var color = "#";
//   for (var i = 0; i < 6; i++) {
//     color += letters[Math.floor(Math.random() * 16)];
//   }
//   return color;
// }

// function getRandomColor(hex, lum) {
//   // validate hex string
//   hex = String(hex).replace(/[^0-9a-f]/gi, "");
//   if (hex.length < 6) {
//     hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
//   }
//   lum = lum || 0;

//   // convert to decimal and change luminosity
//   var rgb = "#",
//     c,
//     i;
//   for (i = 0; i < 3; i++) {
//     c = parseInt(hex.substr(i * 2, 2), 16);
//     c = Math.round(Math.min(Math.max(0, c + c * lum), 255)).toString(16);
//     rgb += ("00" + c).substr(c.length);
//   }

//   return rgb;
// }

export class App extends Component {
  constructor(props) {
    super(props);
    this.initState = {
      loading: false,
      showPrediction: false,
      text: "",
      target: "",
      mode: "auto",
      encoding: "c",
      targetGrouping: "n",
      modelName: "gb",
      prediction: null,
      email: "",
      feedback: "",
      expectedStance: "",
    };
    this.state = { ...this.initState };
  }

  handleSubmit = (e) => {
    e.preventDefault();

    let inputForm = document.getElementById("inputForm");
    let formData = new FormData(inputForm);

    this.setState(
      { loading: true, showPrediction: false, prediction: null },
      () => {
        axiosInstance
          .post("", formData)
          .then(({ data }) => {
            toast.success("Execution complete");
            this.setState({
              loading: false,
              prediction: data,
              showPrediction: true,
            });
          })
          .catch(({ message }) => {
            toast.error("Error encountered");
            this.setState({
              loading: false,
              showPrediction: false,
            });
          });
      }
    );
  };

  handleChange = (e) => {
    this.setState({ [e.target.name]: e.target.value });
  };

  reportIncorrectPrediction = (e) => {
    e.preventDefault();

    const reportIncorrectPredictionForm = document.getElementById(
      "reportIncorrectPrediction"
    );
    const reportIncorrectPredictionFormData = new FormData(
      reportIncorrectPredictionForm
    );

    console.log(reportIncorrectPredictionFormData);
    reportIncorrectPredictionFormData.append(
      "text",
      this.state.prediction.text
    );
    reportIncorrectPredictionFormData.append(
      "target",
      this.state.prediction.target
    );

    document.getElementById("reportIncorrectPredictionClose").click();
    axiosInstance
      .post("reportIncorrectPrediction/", reportIncorrectPredictionFormData)
      .then((res) => {
        toast.success("Successfully reported");
      })
      .catch((err) => {
        toast.error(err.message);
      });
  };

  getOverallStance = () => {
    // majority -> confidence

    const outputs = this.state.prediction["outputs"];
    let count0 = 0;
    let confidence0 = 0;
    let count1 = 0;
    let confidence1 = 0;
    let count2 = 0;
    let confidence2 = 0;

    for (let i = 0; i < outputs.length; i++) {
      const o = outputs[i];
      if (o.stance === "against") {
        count0++;
        confidence0 += o.confidence * 100;
      }
      if (o.stance === "favor") {
        count1++;
        confidence1 += o.confidence * 100;
      }
      if (o.stance === "neutral") {
        count2++;
        confidence2 += o.confidence * 100;
      }
    }
    const d = [count0, count1, count2];
    const dconf = [confidence0, confidence1, confidence2];

    const maxVal = _.max(d);
    const index = d.indexOf(maxVal);

    const maxValConf = _.max(dconf);
    const indexConf = dconf.indexOf(maxValConf);

    // 3-1-1, 3-2, 4-1, 5-0
    if (maxVal === 3 || maxVal === 4 || maxVal === 5) {
      if (index === 0) return "against";
      if (index === 1) return "favor";
      if (index === 2) return "neutral";
    }
    // 2-2-1
    else {
      if (indexConf === 0) return "against";
      if (indexConf === 1) return "favor";
      if (indexConf === 2) return "neutral";
    }
  };

  render() {
    return (
      <div
        id="app"
        className="container-fluid d-flex flex-column"
        // onContextMenu={(e) => e.preventDefault()}
      >
        <header className="d-flex justify-content-center">
          <h1
            className="p-3 animate__animated animate__rubberBand"
            style={{
              textShadow: "0 .2rem 1rem #ddd",
              fontWeight: "bolder",
            }}
          >
            Stance Detection
          </h1>
        </header>

        <main className="col flex-grow-1 d-flex flex-column flex-lg-row">
          {/* input */}
          <section className="col-12 col-lg-6 px-lg-5 animate__animated animate__fast animate__zoomInLeft">
            {this.state.loading ? (
              <div className="d-flex justify-content-center p-5">
                <div className="loader">
                  <svg viewBox="0 0 80 80">
                    <circle id="test" cx="40" cy="40" r="32"></circle>
                  </svg>
                </div>

                <div className="loader triangle">
                  <svg viewBox="0 0 86 80">
                    <polygon points="43 8 79 72 7 72"></polygon>
                  </svg>
                </div>

                <div className="loader">
                  <svg viewBox="0 0 80 80">
                    <rect x="8" y="8" width="64" height="64"></rect>
                  </svg>
                </div>
              </div>
            ) : (
              <form id="inputForm" onSubmit={this.handleSubmit}>
                <div className="mb-3">
                  <label htmlFor="text" className="form-label">
                    Text
                  </label>
                  <input
                    placeholder="E.g. Climate change is a global concern."
                    autoFocus
                    value={this.state.text}
                    onChange={this.handleChange}
                    autoComplete="false"
                    type="text"
                    name="text"
                    id="text"
                    className="form-control"
                    required
                  />
                </div>

                <div className="mb-3">
                  <label htmlFor="target" className="form-label">
                    Target
                  </label>
                  <input
                    placeholder="E.g. Global warming"
                    value={this.state.target}
                    onChange={this.handleChange}
                    autoComplete="false"
                    type="text"
                    name="target"
                    id="target"
                    className="form-control"
                    required
                  />
                </div>

                <div className="mb-3">
                  <label htmlFor="mode" className="form-label">
                    Mode
                  </label>
                  <select
                    name="mode"
                    id="mode"
                    className="form-select"
                    value={this.state.mode}
                    onChange={this.handleChange}
                  >
                    <option defaultValue="auto" value="auto">
                      Auto
                    </option>
                    <option value="manual">Manual</option>
                  </select>
                  <p className="form-text">
                    "Auto" mode returns the result from the best performing
                    model for you. "Manual" mode allows you to customize the
                    model parameters.
                  </p>
                  {/* {this.state.mode === 'auto' && <small>The output in auto mode is calculated as by taking majority of the predicted stances. In case of a tie, we display stance having the maximum confidence.</small>} */}
                </div>

                {this.state.mode === "manual" && (
                  <>
                    <div className="mb-3">
                      <label htmlFor="encoding" className="form-label">
                        Encoding
                      </label>
                      <select
                        value={this.state.encoding}
                        onChange={this.handleChange}
                        name="encoding"
                        id="encoding"
                        className="form-select"
                      >
                        <option defaultValue="c" value="c">
                          Contextual
                        </option>
                        <option value="cf">Context-Free</option>
                      </select>
                      <p className="form-text">
                        Contextual encoding considers a words' semantic meaning
                        in the sentence and uses that to predict the results.
                        Context-Free encoding takes individual word's meaning in
                        isolation and uses them to the predict results.
                      </p>
                    </div>

                    <div className="mb-3">
                      <label htmlFor="targetGrouping" className="form-label">
                        Target Grouping
                      </label>
                      <select
                        value={this.state.targetGrouping}
                        onChange={this.handleChange}
                        name="targetGrouping"
                        id="targetGrouping"
                        className="form-select"
                      >
                        <option defaultValue="n" value="n">
                          None
                        </option>
                        <option value="a">Agglomerative</option>
                        <option value="b">BERTopic</option>
                      </select>
                      <p className="form-text">
                        Target grouping allows the models to use a more
                        generalized representation of the target to predict the
                        results.
                      </p>
                    </div>

                    <div className="mb-3">
                      <label htmlFor="modelName" className="form-label">
                        Machine Learning Model
                      </label>
                      <select
                        name="modelName"
                        id="modelName"
                        className="form-select"
                        value={this.state.modelName}
                        onChange={this.handleChange}
                      >
                        <option value="gb" defaultValue="gb">
                          Gradient Boosting Classifier
                        </option>
                        <option value="svc">Support Vector Classifier</option>
                        <option value="gnb">
                          Gaussian Naive Bayes Classifier
                        </option>
                        <option value="lr">Logistic Regression</option>
                        <option value="dt">Decision Tree Classifier</option>
                        <option value="rf">Random Forest Classifier</option>
                        <option value="knn">
                          K-Nearest Neighbors Classifier
                        </option>
                        <option value="d">
                          Fully Connected Neural Network
                        </option>
                        <option value="l">LSTM Neural Network</option>
                        <option value="b">Bi-LSTM Neural Network</option>
                      </select>
                      <p className="form-text">
                        Choose the Machine Learning/Neural Network model to use
                        to predict the results.
                      </p>
                    </div>
                  </>
                )}

                <div className="mb-3 d-flex justify-content-between">
                  <button type="submit" className="btn btn-primary">
                    Submit
                  </button>
                </div>
              </form>
            )}
            <div className="d-lg-none d-block">
              <hr />
            </div>
          </section>

          {/* output */}
          <section className="col-12 col-lg-6 px-lg-5">
            {this.state.showPrediction ? (
              // prediction
              <>
                <div className="modal fade" id="reportIncorrect" tabIndex="-1">
                  <div className="modal-dialog">
                    <form
                      id="reportIncorrectPrediction"
                      className="modal-content shadow border-0"
                      onSubmit={this.reportIncorrectPrediction}
                      style={{
                        background: "#ffffffcc",
                        borderRadius: "1rem",
                        backdropFilter: "blur(.4rem)",
                      }}
                    >
                      <div className="modal-header">
                        <h5 className="modal-title" id="reportIncorrectLabel">
                          Report Incorrect Prediction
                        </h5>
                        <button
                          id="reportIncorrectPredictionClose"
                          type="button"
                          className="btn-close"
                          data-bs-dismiss="modal"
                        ></button>
                      </div>
                      <div className="modal-body">
                        <div className="mb-3">
                          <input
                            value={this.state.email}
                            onChange={this.handleChange}
                            type="email"
                            name="email"
                            id="email"
                            className="form-control"
                            placeholder="E-mail address"
                            required
                          />
                        </div>
                        <div className="mb-3">
                          <input
                            value={this.state.feedback}
                            onChange={this.handleChange}
                            type="text"
                            name="feedback"
                            id="feedback"
                            className="form-control"
                            placeholder="Optional feedback message"
                          />
                        </div>
                        <div className="mb-3">
                          <label
                            htmlFor="expectedStance"
                            className="form-label"
                          >
                            Expected Stance
                          </label>
                          <select
                            value={this.state.expectedStance}
                            onChange={this.handleChange}
                            name="expectedStance"
                            id="expectedStance"
                            className="form-select"
                          >
                            <option defaultValue="favor" value="favor">
                              Favor
                            </option>
                            <option value="against">Against</option>
                            <option value="neutral">Neutral</option>
                          </select>
                          <p className="form-text"></p>
                        </div>
                        <div className="mb-3" id="captcha"></div>
                      </div>
                      <div className="modal-footer">
                        <button type="submit" className="btn btn-primary">
                          Report
                        </button>
                      </div>
                    </form>
                  </div>
                </div>
                {/* panel */}
                <>
                  <div
                    className="bg-light p-3 mb-3"
                    style={{
                      borderRadius: "1rem",
                    }}
                  >
                    <p>
                      <p className="fw-bold fs-5">Text</p>
                      <span className="text-secondary">
                        {this.state.prediction && this.state.prediction.text}
                      </span>
                    </p>
                    <p>
                      <p className="fw-bold fs-5">Target</p>
                      <span className="text-secondary">
                        {this.state.prediction && this.state.prediction.target}
                      </span>
                    </p>
                  </div>
                  {/* <p>
                    <b>Stance: </b>
                    <span
                      className={
                        this.state.prediction &&
                        (this.state.prediction.stance === "favor"
                          ? "text-white p-2 m-1 shadow-sm bg-success"
                          : this.state.prediction.stance === "against"
                          ? "text-white p-2 m-1 shadow-sm bg-danger"
                          : "text-white p-2 m-1 shadow-sm bg-secondary")
                      }
                    >
                      {this.state.prediction &&
                        this.state.prediction.stance.toUpperCase()}
                    </span>
                  </p> */}

                  {/* <p
                    className="btn-link mb-3"
                    role="button"
                    data-bs-toggle="collapse"
                    data-bs-target="#predictionInfo"
                  >
                    More details
                  </p> */}
                  {this.state.prediction &&
                  this.state.prediction.mode === "manual" ? (
                    <div
                      className="collaps bg- shadow-sm text-white mb-3 d-flex justify-content-between align-items-center p-3"
                      id="predictionInfo"
                      style={{ borderRadius: "1rem", backgroundColor: "#666" }}
                    >
                      <div>
                        <p className="fw-bold fs-5">Model</p>
                        <span className="text-  ">
                          {this.state.prediction &&
                            this.state.prediction.modelInfo.modelName}
                        </span>
                      </div>
                      <div className="output__progressbar text-center">
                        <CircularProgressbar
                          className="w-50"
                          value={
                            this.state.prediction &&
                            this.state.prediction.modelInfo.confidence
                          }
                          maxValue={1}
                          text={
                            this.state.prediction &&
                            voca.capitalize(this.state.prediction.stance)
                          }
                          strokeWidth={12}
                          styles={buildStyles({
                            textSize: ".7rem",
                            textColor: "white",
                            pathColor:
                              this.state.prediction &&
                              (this.state.prediction.stance === "favor"
                                ? "green"
                                : this.state.prediction.stance === "against"
                                ? "red"
                                : "grey"),
                          })}
                        />
                      </div>
                    </div>
                  ) : this.state.prediction ? (
                    <>
                      <div
                        className="collaps bg- shadow-sm text-white mb-2 d-flex justify-content-between align-items-center p-3"
                        id="predictionInfo"
                        style={{
                          borderRadius: "1rem",
                          // backgroundColor: getRandomColor(),
                          backgroundColor: "#333",
                        }}
                      >
                        <div className="col-8">
                          {/* <p className="fw-bold fs-5">Model</p> */}
                          <span className="text-  ">{`Overall stance`}</span>
                        </div>
                        <div className="output__progressbar text-center d-">
                          {voca.capitalize(this.getOverallStance())}
                        </div>
                      </div>

                      <div className="accordion my-3" id="automodeoutputs">
                        <div className="accordion-item">
                          <h2 className="accordion-header">
                            <button
                              className="accordion-button"
                              type="button"
                              data-bs-toggle="collapse"
                              data-bs-target="#howoutputispredicted"
                            >
                              How the output is predicted?
                            </button>
                          </h2>
                          <div
                            id="howoutputispredicted"
                            className="accordion-collapse collapse show"
                            data-bs-parent="#automodeoutputs"
                          >
                            <div className="accordion-body">
                              <p>
                                The output in auto mode is calculated as by
                                taking majority of the predicted stances. In
                                case of a tie, we display stance having the
                                maximum confidence.
                              </p>
                            </div>
                          </div>
                        </div>
                        <div className="accordion-item">
                          <h2 className="accordion-header">
                            <button
                              className="accordion-button collapsed"
                              type="button"
                              data-bs-toggle="collapse"
                              data-bs-target="#individualmodelspredictions"
                            >
                              Individual models' prediction
                            </button>
                          </h2>
                          <div
                            id="individualmodelspredictions"
                            className="accordion-collapse collapse"
                            data-bs-parent="#automodeoutputs"
                          >
                            <div className="accordion-body">
                              <div className="" id="individual__predicitions">
                                {this.state.prediction["outputs"].map(
                                  (output) => (
                                    <div
                                      className="collaps bg- shadow-sm text-white mb-3 d-flex justify-content-between align-items-center p-3"
                                      id="predictionInfo"
                                      style={{
                                        borderRadius: "1rem",
                                        // backgroundColor: getRandomColor(),
                                        backgroundColor: "#333",
                                      }}
                                      // style={{
                                      //   borderRadius: "1rem",
                                      // background: `linear-gradient(to right, ${
                                      //   output.stance === "favor"
                                      //     ? "green"
                                      //     : output.stance === "against"
                                      //     ? "red"
                                      //     : "grey"
                                      // } ${output.confidence * 100}%, white)`,
                                      // }}
                                    >
                                      <div className="col-8">
                                        {/* <p className="fw-bold fs-5">Model</p> */}
                                        <span className="text-  ">
                                          {output.modelName}
                                        </span>
                                      </div>
                                      <div
                                        className="output__progressbar text-center d-"
                                        // style={{
                                        //   width: 200,
                                        //   height: 200,
                                        // }}
                                      >
                                        <CircularProgressbar
                                          className="w-50"
                                          value={output.confidence}
                                          maxValue={1}
                                          text={voca.capitalize(output.stance)}
                                          strokeWidth={12}
                                          styles={buildStyles({
                                            textSize: ".7rem",
                                            textColor: "white",
                                            pathColor:
                                              output.stance === "favor"
                                                ? "green"
                                                : output.stance === "against"
                                                ? "red"
                                                : "grey",
                                          })}
                                        />
                                      </div>
                                    </div>
                                  )
                                )}
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    </>
                  ) : null}

                  <div className="d-flex justify-content-between mb-3">
                    <button
                      className={
                        this.state.prediction.mode === "manual"
                          ? "btn btn-dark"
                          : "d-none"
                      }
                      type="button"
                      data-bs-toggle="modal"
                      data-bs-target="#reportIncorrect"
                    >
                      Report incorrect prediction
                    </button>
                    {/* <button className="btn btn-secondary">Save</button> */}
                    <button
                      className="btn btn-danger"
                      onClick={() => {
                        this.setState({ ...this.initState });
                      }}
                    >
                      Reset
                    </button>
                  </div>
                </>
              </>
            ) : (
              // information screen
              <div className="animate__animated animate__fast animate__zoomInRight">
                <p>
                  Stance is defined as the expression of the speaker's
                  standpoint and judgement toward a given proposition.
                </p>
                <p>
                  Stance detection is the task of automatically determining from
                  a text whether the author of the text is in favor of, against,
                  or neutral towards a target.
                </p>

                <p>
                  The models used for the prediction of stance have been trained
                  on the <b>VAST (VAried Stance Topics)</b> dataset.
                </p>
                <p>
                  We preferred VAST dataset over other existing datasets on the
                  Internet. As the existing datasets fall short because they
                  have handful number of targets, unclear or no explicit labels,
                  no linguistic variations in target expressions - to mention a
                  few.
                </p>
                <p>
                  The VAST training dataset has around 13,000 few-shot and
                  zero-shot entries.
                </p>
                <img
                  src={trainingDataset}
                  alt="Training data"
                  className="img-fluid"
                />
                <p>
                  The distribution of the labels is uneven in the training
                  dataset. Ergo, we adjusted the class weights during model
                  fitting to get better predictions of neutral stances.
                </p>
                <p>
                  Our solution gives support for large number of targets by
                  generalizing the targets to better predict the results.
                </p>
                <img
                  src={modelWorking}
                  alt="Stance Detection working"
                  className="img-fluid mb-3"
                />
                {/* <p>
                  <a href="" target="_blank" rel="noopener noreferrer">
                    Models' Information
                  </a>
                </p> */}
              </div>
            )}
          </section>
        </main>
      </div>
    );
  }
}

export default App;
