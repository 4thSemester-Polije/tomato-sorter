import { GRADE_NOTE } from "../format.js";

export default function GradeChip({ grade, big }) {
  return (
    <span className={`grade grade-${grade}${big ? " big" : ""}`} title={GRADE_NOTE[grade]} aria-label={`Grade ${grade}`}>
      {grade}
    </span>
  );
}
