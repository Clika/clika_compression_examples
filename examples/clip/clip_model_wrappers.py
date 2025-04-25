from typing import Optional
import torch
from transformers import CLIPModel


class CLIPVisionModel(torch.nn.Module):

    def __init__(self, model: CLIPModel):
        super().__init__()
        self.model = model

    def forward(self,
                pixel_values: Optional[torch.FloatTensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                interpolate_pos_encoding: bool = False,
                ):
        return self.model.get_image_features(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )

class CLIPTextModel(torch.nn.Module):

    def __init__(self, model: CLIPModel):
        super().__init__()
        self.model = model

    def forward(self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                ):
        return self.model.get_text_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
