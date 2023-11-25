import copy

from cues import constants, cursor, utils, Survey


class RegelumSurvey(Survey):
    def _draw(self):
        """Print the prompt to console and set user's response."""
        cursor.write(self._init_fmt.format(msg=self._message), color=True)

        if self._legend:
            # If there are only two elems in self._legend:
            if self._header_fmt:
                cursor.write(
                    self._header_fmt.format(
                        *self._legend, space=self._total_legend_fmt_len * " "
                    ),
                    color=True,
                    newlines=1,
                )
                cursor.write(
                    self._legend_fmt.format(*self._scale, space=self._space_btwn),
                    color=True,
                    newlines=3,
                )
            # else, if lengths of _legend and _scale are equal:
            else:
                for pt, desc in zip(self._scale, self._legend):
                    cursor.write(
                        self._legend_fmt.format(val=pt, legend=desc), color=True
                    )
                cursor.write("", newlines=2)

        # For keeping track of location:
        scale_len = len(self._scale)
        max_fields = len(self._fields) * 4
        current_field = max_fields

        # Creates line between survey points:
        deque_scale = self.create_deque(self._scale)
        current_deque_scale = copy.copy(deque_scale)

        center_pt = utils.get_half(scale_len)
        pts = [constants.SURVEY_PT for _ in range(scale_len)]
        pts[center_pt - 1] = constants.SURVEY_PT_FILL
        deque_pts = self.create_deque(pts)
        current_deque_pts = copy.copy(deque_pts)

        min_space_btwn_lines = 5
        max_line_len = max(len(elem) for elem in deque_scale) + min_space_btwn_lines
        line = constants.SURVEY_LINE * max_line_len

        scale_str = ""
        for val in deque_scale:
            scale_str += self._scale_fmt.format(val, length=max_line_len + 1)
        scale_str += "\n"

        messages = [field["message"] for field in self._fields]

        default_margin = 2

        for c, message in enumerate(messages, 1):
            cursor.write(self._msg_fmt.format(count=c, msg=message))

            # Adds space in front:
            margin = " " * (default_margin + utils.get_num_digits(c))

            cursor.write(
                margin + self._pt_fmt.format(*deque_pts, line=line), color=True
            )
            cursor.write(margin + scale_str + "\n")

        cursor.move(y=current_field)

        right = self.keys.get("right")
        left = self.keys.get("left")
        enter = self.keys.get("enter")

        horziontal_num = center_pt
        current_val = 0

        responses = {}

        # Actual drawing:
        while True:
            if current_val == 1:
                break
            cursor.write(
                self._msg_fmt.format(count=current_val + 1, msg=messages[current_val])
            )

            # Adds space in front:
            margin = " " * (default_margin + utils.get_num_digits(current_val + 1))

            cursor.write(
                margin + self._pt_fmt.format(*current_deque_pts, line=line), color=True
            )

            scale_str = ""
            for c, val in enumerate(current_deque_scale, 1):
                temp_line_len = 0
                if c == horziontal_num:
                    val = (
                        "[underline lightslateblue]"
                        + val
                        + "[/underline lightslateblue]"
                    )
                    temp_line_len = max_line_len + len(val)
                scale_str += self._scale_fmt.format(
                    val, length=(temp_line_len or max_line_len + 1)
                )
            scale_str += "\n"
            cursor.write(margin + scale_str + "\n", color=True)

            cursor.move(y=-(current_field - 4))

            key = self.listen_for_key()

            if key == right:
                # If cursor is at very right:
                if current_deque_pts[-1] == constants.SURVEY_PT_FILL:
                    pass
                else:
                    current_deque_pts.appendleft(constants.SURVEY_PT)
                    horziontal_num += 1

            elif key == left:
                # If cursor is at very left:
                if current_deque_pts[0] == constants.SURVEY_PT_FILL:
                    pass
                else:
                    current_deque_pts.append(constants.SURVEY_PT)
                    horziontal_num -= 1

            elif key == enter:
                # Add current scale value to dict
                responses.update(
                    {self._fields[current_val]["name"]: self._scale[horziontal_num - 1]}
                )
                current_val += 1

                # If at the end of the survey, then quit:
                if current_val == scale_len - 1:
                    if self._header_fmt:
                        cursor.clear(max_fields + 4)
                    else:
                        cursor.clear(
                            max_fields + (len(self._legend) + 2 if self._legend else 0)
                        )

                    break
                else:
                    current_field -= 4
                    # Resets values:
                    current_deque_pts = copy.copy(deque_pts)
                    current_deque_scale = copy.copy(deque_scale)
                    horziontal_num = center_pt

            # Resets cursor at top:
            cursor.move(y=current_field)

        self.answer = {self._name: responses}
