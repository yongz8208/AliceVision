#include "timeLine.hpp"

namespace aliceVision {
namespace sfmData {

bool TimeLine::addView(IndexT id, uint64_t timestamp) 
{
    TimedMeasure mes;
    mes.id = id;
    mes.timeCode = timestamp;
    mes.timedMeasureType = TimedMeasure::TimedMeasureTypeView;

    _measures[id] = mes;

    return true;
}

} // namespace sfmData
} // namespace aliceVision