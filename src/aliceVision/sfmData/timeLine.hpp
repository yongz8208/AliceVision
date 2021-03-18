#pragma once

#include <aliceVision/types.hpp>

namespace aliceVision {
namespace sfmData {

struct TimedMeasure
{
    enum TimedMeasureType
    {
        TimedMeasureTypeView,
        TimedMeasureTypeOther
    };
    
    TimedMeasureType timedMeasureType;
    uint64_t timeCode;
    IndexT id;
};

class TimeLine
{
public:
    bool addView(IndexT id, uint64_t timestamp);

    const std::map<uint64_t, TimedMeasure> & getMeasures() const
    {
        return _measures;
    }
    
private:
    std::map<uint64_t, TimedMeasure> _measures;
};

} // namespace sfmData
} // namespace aliceVision