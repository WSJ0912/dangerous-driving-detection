# -*- coding: utf-8 -*-
"""行为分析与告警（含冷却窗口和事件生命周期管理）

参考业界 DMS (Driver Monitoring System) 项目标准:
- 事件首次达到阈值 -> 触发告警 (triggered)
- 事件持续过程中 -> 不重复告警 (ongoing)
- 事件结束后 --> 发出解除通知 (resolved)
- 冷却窗口内同一类型不再触发新事件
"""
from backend import config


class AlertEvent:
    """单次告警事件，含状态生命周期"""
    __slots__ = ('type', 'status', 'message', 'frame_count')

    def __init__(self, alert_type, status, message, frame_count=0):
        self.type = alert_type
        self.status = status  # triggered / ongoing / resolved
        self.message = message
        self.frame_count = frame_count

    def to_dict(self):
        return {
            'type': self.type,
            'status': self.status,
            'message': self.message,
            'frame_count': self.frame_count,
        }


class BehaviorAnalyzer:
    """逐帧累加分析驾驶行为状态，含冷却窗口和事件生命周期"""

    def __init__(self, thresholds=None, cooldown_frames=30):
        """Args:
            thresholds: 告警阈值 dict
            cooldown_frames: 事件解除后冷却帧数
        """
        self._thresholds = thresholds or config.WARNING_THRESHOLDS
        self._cooldown = cooldown_frames
        self.state = {
            'eye_status': 'open',
            'yawn_status': 'noYawn',
            'distracted': False,
            'eye_close_frames': 0,
            'yawn_frames': 0,
            'distracted_frames': 0,
        }
        self.alert_state = {'eye_close': False, 'yawn': False, 'distracted': False}
        self.active_events = {}  # {type: AlertEvent}
        self._cooldown_counters = {}  # {type: remaining_frames}

    def analyze(self, detections):
        """分析当前帧检测结果，返回新产生的事件列表

        Returns:
            list[AlertEvent]: 只返回 triggered 和 resolved 事件
        """
        self._tick_cooldown()
        self._extract_current_state(detections)
        self._update_alert_state()

        events = []
        events += self._process_alert_type(
            'eye_close', self.alert_state['eye_close'],
            config.ALERT_MESSAGES['eye_close'])
        events += self._process_alert_type(
            'yawn', self.alert_state['yawn'],
            config.ALERT_MESSAGES['yawn'])
        events += self._process_alert_type(
            'distracted', self.alert_state['distracted'],
            config.ALERT_MESSAGES['distracted'])
        return events

    def _tick_cooldown(self):
        """推进冷却计数器"""
        expired = [k for k, v in self._cooldown_counters.items() if v <= 1]
        for key in expired:
            del self._cooldown_counters[key]
        for key in list(self._cooldown_counters.keys()):
            self._cooldown_counters[key] -= 1

    def _extract_current_state(self, detections):
        current_eye = 'open'
        current_yawn = 'noYawn'
        current_distracted = False
        for det in detections:
            cls = det['class']
            if cls == 'close_eye':
                current_eye = 'close'
            elif cls == 'open_eye':
                current_eye = 'open'
            elif cls == 'Yawn':
                current_yawn = 'Yawn'
            elif cls == 'noYawn':
                current_yawn = 'noYawn'
            elif cls in ('phone', 'smoke', 'drink'):
                current_distracted = True
        self._update_counter('eye_close_frames', current_eye == 'close')
        self.state['eye_status'] = current_eye
        self._update_counter('yawn_frames', current_yawn == 'Yawn')
        self.state['yawn_status'] = current_yawn
        self._update_counter('distracted_frames', current_distracted)
        self.state['distracted'] = current_distracted

    def _update_alert_state(self):
        self.alert_state['eye_close'] = (
            self.state['eye_close_frames'] >= self._thresholds['eye_close'])
        self.alert_state['yawn'] = (
            self.state['yawn_frames'] >= self._thresholds['yawn'])
        self.alert_state['distracted'] = (
            self.state['distracted_frames'] >= self._thresholds['distracted'])

    def _update_counter(self, key, condition):
        if condition:
            self.state[key] += 1
        else:
            self.state[key] = 0

    def _process_alert_type(self, key, is_active, message):
        events = []
        in_cooldown = key in self._cooldown_counters

        if is_active and not in_cooldown:
            if key not in self.active_events:
                # 首次触发
                fc = self.state.get(f'{key}_frames', 1)
                evt = AlertEvent(key, 'triggered', message, fc)
                self.active_events[key] = evt
                events.append(evt)
            else:
                # 持续中，更新帧数，不产生新事件
                self.active_events[key].status = 'ongoing'
                self.active_events[key].frame_count = self.state.get(f'{key}_frames', 1)
        elif not is_active and key in self.active_events:
            # 事件解除
            evt = AlertEvent(
                key, 'resolved',
                f'{self.active_events[key].message}（已解除）',
                self.active_events[key].frame_count)
            events.append(evt)
            del self.active_events[key]
            self._cooldown_counters[key] = self._cooldown

        return events
