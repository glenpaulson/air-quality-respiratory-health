import ModelScaffold from "../components/ModelScaffold";

const AdaBoost = () => (
  <ModelScaffold
    title="AdaBoost"
    module="Module 4"
    plan="AdaBoost will combine many shallow trees into one stronger classifier, giving extra weight to the days it keeps getting wrong, which should help with the rare unhealthy days that the other methods tend to miss."
  />
);

export default AdaBoost;
