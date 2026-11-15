import {describe,it,expect} from 'vitest';
import {timeLabel,parseTime} from './api';
describe('recording-relative timestamps',()=>{
  it('preserves hours beyond midnight',()=>expect(timeLabel(90061)).toBe('25:01:01'));
  it('parses precise boundaries',()=>expect(parseTime('01:02:03.5')).toBe(3723.5));
  it('does not accept invalid minutes or negative offsets',()=>{expect(parseTime('00:99:00')).toBeNull();expect(parseTime('-1')).toBeNull();expect(parseTime('')).toBeNull()});
  it('accepts elapsed seconds',()=>expect(parseTime('728.5')).toBe(728.5));
});
