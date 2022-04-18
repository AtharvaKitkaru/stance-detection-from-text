import axios from "axios";

const axiosInstance = axios.create({
  baseURL: process.env.REACT_APP_API,
  xsrfHeaderName: "HTTP_X_CSRFTOKEN",
  xsrfCookieName: "csrftoken",
});

export default axiosInstance;
