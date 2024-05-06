import torch
import os
import torch.nn as nn
import sys
from torchvision import transforms

# add the project path to the system path
project_absolute_path_from_loss_py = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# add the project path to the system path
sys.path.append(project_absolute_path_from_loss_py)

# import the function to download the VGG19 model and create the cutted model
from codes.utils import get_scaled_self_cosine_distance_map_lower_triangle

# define the custom VGG19 model using the original VGG19 model as input
class VGG19_custom(nn.Module):
    def __init__(self, features: nn.Module):
        super().__init__()

        # set the features (list of layers of the VGG19 model)
        self.features = features

    # define the forward function
    def forward(self, x):
        # get the output from relu 2_1
        relu_2_1_output = self.features[:7](x)

        # get the output from relu 3_1
        relu_3_1_output = self.features[7:12](relu_2_1_output)

        # get the output from relu 4_1
        relu_4_1_output = self.features[12:21](relu_3_1_output)

        # get the output from relu 5_1
        relu_5_1_output = self.features[21:30](relu_4_1_output)

        # return the outputs as a list
        return [relu_2_1_output, relu_3_1_output, relu_4_1_output, relu_5_1_output]
    

# define the custom VGG19 model using the original VGG19 model with batchnorm as input 
class VGG19_custom_with_batch_norm(nn.Module):
    def __init__(self, features: nn.Module):
        super().__init__()

        # set the features (list of layers of the VGG19 model)
        self.features = features

    # define the forward function
    def forward(self, x):
        # get the output from relu 2_1
        relu_2_1_output = self.features[:11](x)

        # get the output from relu 3_1
        relu_3_1_output = self.features[11:18](relu_2_1_output)

        # get the output from relu 4_1
        relu_4_1_output = self.features[18:31](relu_3_1_output)

        # get the output from relu 5_1
        relu_5_1_output = self.features[31:44](relu_4_1_output)

        # return the outputs as a list
        return [relu_2_1_output, relu_3_1_output, relu_4_1_output, relu_5_1_output]






# construct the loss class
class custom_loss(nn.Module):
    """
    When this class is initialized, it loads the custom VGG19 model, which is cutted at the last layer of relu 5_1.
    If this cutted model is not saved, it downloads the original VGG19 model and creates the cutted model.
    The class calculates the total loss (content loss + lambda * style loss) for the output image, content image, and style image.
    """
    def __init__(self,
                 project_absolute_path,
                 feature_extractor_model_relative_path="weights/vgg_19_last_layer_is_relu_5_1_output.pt",
                 use_vgg19_with_batchnorm=False,
                 default_lambda_value=10,):
        super().__init__()

        # set the lambda value
        self.lambda_value = default_lambda_value

            
        # get the absolute path of the feature extractor model
        feature_extractor_model_path = os.path.join(project_absolute_path, feature_extractor_model_relative_path)

        if use_vgg19_with_batchnorm:
            # change the path to the model with batchnorm
            feature_extractor_model_path = feature_extractor_model_path.replace(".pt", "_bn.pt")

        # check if the VGG19 model is created and saved
        if not os.path.exists(feature_extractor_model_path):
            # add the project path to the system path
            import sys
            sys.path.append(project_absolute_path)

            # import the function to download the VGG19 model and create the cutted model
            from codes.utils import download_VGG19_and_create_cutted_model_to_process

            # create the VGG19 cutted model and save it
            download_VGG19_and_create_cutted_model_to_process(project_absolute_path,
                                                              feature_extractor_model_relative_path,
                                                              use_vgg19_with_batchnorm=use_vgg19_with_batchnorm)

        if use_vgg19_with_batchnorm:
            # load the custom VGG19 model with batchnorm
            self.feature_extractor_model = VGG19_custom_with_batch_norm(torch.load(feature_extractor_model_path))
        else:
            # load the custom VGG19 model without batchnorm
            self.feature_extractor_model = VGG19_custom(torch.load(feature_extractor_model_path))


        # set the model to evaluation mode
        self.feature_extractor_model.eval()

        # freeze the model
        for param in self.feature_extractor_model.parameters():
            param.requires_grad = False

    # define the forward function
    def forward(self,
                content_image,
                style_image,
                output_image,
                output_content_and_style_loss=False,
                output_similarity_loss=False):
        """
        Gets the content image, style image, and output image, and returns the total loss (content loss + lambda * style loss)
        All images should be in the exact same shape: [batch_size, 3, 256, 256]
        """
        return self.get_overall_loss(content_image = content_image,
                                     style_image = style_image,
                                     output_image = output_image,
                                     loss_weight = self.lambda_value,
                                     output_content_and_style_loss = output_content_and_style_loss,
                                     output_similarity_loss = output_similarity_loss)
    


    # Overall, weighted loss (containin both content and style loss)
    def get_overall_loss(self,
                         content_image,
                         style_image,
                         output_image,
                         loss_weight=None,
                         output_content_and_style_loss=False,
                         output_similarity_loss=False):
        """
        This function calculates the total loss (content loss + lambda * style loss) for the output image.
        It uses the custom VGG19 model to get the outputs from relu 2_1, relu 3_1, relu 4_1, relu 5_1 layers, as it is declared in the paper.
        """
        # inputs are in shape: [batch_size, 3, 256, 256]

        # check if lambda value is given
        if loss_weight is None:
            loss_weight = self.lambda_value

        # get the VGG features for content, style, and output images
        VGG_features_content = self.feature_extractor_model(content_image) 
        VGG_features_style = self.feature_extractor_model(style_image)
        VGG_features_output = self.feature_extractor_model(output_image)

        # all above are lists with 4 elements
        # first element of each list is the output from relu 2_1 layer,  which is in shape: [batch_size, 128, 128, 128]
        # second element of each list is the output from relu 3_1 layer, which is in shape: [batch_size, 256, 64, 64]
        # third element of each list is the output from relu 4_1 layer,  which is in shape: [batch_size, 512, 32, 32]
        # fourth element of each list is the output from relu 5_1 layer, which is in shape: [batch_size, 512, 16, 16]

        # calculate losses
        content_loss = self.get_content_loss(VGG_features_content, VGG_features_output)
        style_loss = self.get_style_loss(VGG_features_style, VGG_features_output)

        
        # calculate total loss
        total_loss = content_loss + loss_weight * style_loss

        if output_similarity_loss:
            # calculate similarity loss (passing only relu 4_1 and relu 5_1 layers)
            similarity_loss = self.get_similarity_loss(VGG_features_content[-2:], VGG_features_output[-2:])

            # if requested, return the content and style loss too
            if output_content_and_style_loss:
                return total_loss, content_loss, style_loss, similarity_loss
            else:
                # return only the total loss
                return total_loss, similarity_loss
            
        else:
            # if requested, return the content and style loss too
            if output_content_and_style_loss:
                return total_loss, content_loss, style_loss
            else:
                # return only the total loss
                return total_loss
    


    # Content Loss
    def get_content_loss(self, VGG_features_content, VGG_features_output):
        """
        calculates the content loss (normalized perceptual loss in <https://arxiv.org/pdf/1603.08155>)

        NOTE: Originally, in the paper cited above, the loss is scaled by W,H,C and euclidian distance is used.
        In the master paper, the loss is ambiguous to be squared distance or euclidian distance.
        Also, it is not explicitly mentioned that the loss is scaled by W,H,C.
        We assumed the loss is squared distance, and scaled by B,W,H,C (by taking mean instead of sum) as it produced closed loss values reported in the paper.

        inputs:
            VGG_features_content: list of 4 tensors, each tensor is the output of relu 2_1, relu 3_1, relu 4_1, relu 5_1 layers from the content image
            VGG_features_output: list of 4 tensors, each tensor is the output of relu 2_1, relu 3_1, relu 4_1, relu 5_1 layers from the output image
        """

        # define content loss for each term
        content_loss_each_term = lambda A1, A2, instance_norm: torch.mean(torch.square(torch.sub(instance_norm(A1), instance_norm(A2))))

        # calculate content loss for relu 2_1, relu 3_1, relu 4_1, relu 5_1 (also scaled by W,H,C, as in the mentioned paper)
        content_loss =  content_loss_each_term(VGG_features_content[0], VGG_features_output[0], nn.InstanceNorm2d(128)) + \
                        content_loss_each_term(VGG_features_content[1], VGG_features_output[1], nn.InstanceNorm2d(256)) + \
                        content_loss_each_term(VGG_features_content[2], VGG_features_output[2], nn.InstanceNorm2d(512)) + \
                        content_loss_each_term(VGG_features_content[3], VGG_features_output[3], nn.InstanceNorm2d(512))
        

        return content_loss

    # Style Loss
    def get_style_loss(self, VGG_features_style, VGG_features_output):
        """
        calculates the style loss (mean-variance loss in <https://ieeexplore.ieee.org/document/8237429>)

        NOTE: Again, the loss is ambiguous to be squared distance or euclidian distance.
        Also, it is not explicitly mentioned that the loss is scaled by B,W.
        We assumed the loss is squared distance, and scaled by B,W (by taking mean instead of sum) as it produced closed loss values reported in the paper.


        inputs:
            VGG_features_style: list of 4 tensors, each tensor is the output of relu 2_1, relu 3_1, relu 4_1, relu 5_1 layers from the content image
            VGG_features_output: list of 4 tensors, each tensor is the output of relu 2_1, relu 3_1, relu 4_1, relu 5_1 layers from the output image
        """

        # define style loss for each term
        style_loss_each_term = lambda A1, A2 : torch.mean(torch.square(torch.sub(A1.mean([2,3]), A2.mean([2,3])))) + \
                                               torch.mean(torch.square(torch.sub(A1.std([2,3]), A2.std([2,3]))))



        # calculate style loss for relu 2_1, relu 3_1, relu 4_1, relu 5_1
        style_loss =    style_loss_each_term(VGG_features_style[0], VGG_features_output[0]) + \
                        style_loss_each_term(VGG_features_style[1], VGG_features_output[1]) + \
                        style_loss_each_term(VGG_features_style[2], VGG_features_output[2]) + \
                        style_loss_each_term(VGG_features_style[3], VGG_features_output[3])
        return style_loss



    
    # Similarity Loss
    def get_similarity_loss(self, VGG_features_content, VGG_features_output):
        """
        calculates the similarity loss defined in the paper

        inputs:
            VGG_features_content: list of 2 tensors, each tensor is the output of relu 4_1, relu 5_1 layers from the content image
            VGG_features_output: list of 2 tensors, each tensor is the output of relu 4_1, relu 5_1 layers from the output image
        """
        
        # get the scaled self cosine distance map for the content and output images
        scaled_self_cosine_distance_map_content_relu_4_1 = get_scaled_self_cosine_distance_map_lower_triangle(VGG_features_content[0])
        scaled_self_cosine_distance_map_output_relu_4_1 = get_scaled_self_cosine_distance_map_lower_triangle(VGG_features_output[0])
        scaled_self_cosine_distance_map_content_relu_5_1 = get_scaled_self_cosine_distance_map_lower_triangle(VGG_features_content[1])
        scaled_self_cosine_distance_map_output_relu_5_1 = get_scaled_self_cosine_distance_map_lower_triangle(VGG_features_output[1])


        # get the sum of absolute difference between the two matrices
        abs_dif_self_cos_maps_relu_4_1 = torch.sum(torch.abs(torch.sub(scaled_self_cosine_distance_map_content_relu_4_1,
                                                                       scaled_self_cosine_distance_map_output_relu_4_1)))
        
        abs_dif_self_cos_maps_relu_5_1 = torch.sum(torch.abs(torch.sub(scaled_self_cosine_distance_map_content_relu_5_1,
                                                                       scaled_self_cosine_distance_map_output_relu_5_1)))


        # divide with {n_{x}}^{2}, as well as batch sizes
        similarity_loss_relu_4_1 = abs_dif_self_cos_maps_relu_4_1 / (scaled_self_cosine_distance_map_output_relu_4_1.shape[-1] * scaled_self_cosine_distance_map_output_relu_4_1.shape[0])
        similarity_loss_relu_5_1 = abs_dif_self_cos_maps_relu_5_1 / (scaled_self_cosine_distance_map_output_relu_5_1.shape[-1] * scaled_self_cosine_distance_map_output_relu_5_1.shape[0])


        # calculate the similarity loss
        similarity_loss = similarity_loss_relu_4_1 + similarity_loss_relu_5_1


        return similarity_loss
