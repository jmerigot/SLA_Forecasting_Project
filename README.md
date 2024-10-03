# Sea Level Anomaly Forecasting in the Gulf Stream using Neural Network Approaches

## Author: Jules Merigot

*This research project was completed during an end-of-study internship at the LOCEAN laboratory of the IPSL-CNRS collaboration located on the Pierre et Marie Curie Campus of Sorbonne University in Paris. The final report was submitted to the MIDO Department of Computer Science of Paris Dauphine University in partial fulfillment of the requirements for the masters degree of Master IASD, Artificial Intelligence and Data Science.*

---

### Abstract

Sea Level Anomaly (SLA) is an indicator of the sub-mesoscale dynamics of the upper ocean, and reveals the regional extent of anomalous water levels in the ocean which can indicate unusual water temperatures, salinities, and currents. In this study, we focused on the temporal evolution of SLA fields, complimented with Sea Surface Temperature (SST) data. SST is driven by these ocean dynamics and can be used to improve the spatial interpolation of SLA fields. We specifically explored the potential of Deep Learning (DL) solutions with Attention-based methods to forecast short-term SLA fields using SST fields as a complimentary indicator of the SLA dynamics. Our work serves as a proof of concept, therefore we worked with simulated daily SLA and SST data from the Mercator Global Analysis and Forecasting System, with a resolution of $\frac{1}{12}^\circ$ in the North Atlantic Ocean ($26.5-44.42^\circ$ N, - $64.25-41.83^\circ$ E), covering the period from 1993 to 2019. Using a modified image-to-image convolutional DL architecture with attention-based modules, we demonstrated that SST is a relevant variable for controlling the SLA prediction, and managed to improve the SLA forecast at 5 days by using the SST fields as additional information. 

*Keywords*---Deep-learning, Sea Surface Temperature, Forecast, Attention, Transformers.
