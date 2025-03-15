import torch
import torch.nn.functional as F

def dandr_loss(logits_student, logits_teacher, target, alpha, beta, temperature, detach_target=True):
    """
    DANDR loss for distillation with foreground and unknown classes.
    """
    if detach_target:
        logits_teacher = logits_teacher.detach()

    # Foreground and unknown class indices
    index_fg = (target != (logits_teacher.shape[-1] - 1))  # 前景类别 (非 unknown)
    index_unk = (target == (logits_teacher.shape[-1] - 1))  # Unknown 类别

    # Temperature scaling
    logits_teacher = logits_teacher / temperature
    logits_student = logits_student / temperature

    # # Confidence of teacher model output
    # confidence = teacher_probs.max(dim=-1)[0]
    # # If confidence is below the threshold, treat it as an unknown class
    # if confidence < 0.2:
    #     teacher_probs = torch.full_like(teacher_probs, 1 / teacher_probs.size(-1))  # Smooth out predictions

    # Softmax probabilities
    teacher_probs = torch.softmax(logits_teacher, dim=-1)
    student_probs = torch.log_softmax(logits_student, dim=-1)

    # KL divergence loss
    kd_loss = torch.nn.functional.kl_div(student_probs, teacher_probs, reduction='none')
    kd_loss = kd_loss.sum(dim=-1) * (temperature ** 2)  # KL divergence scaling

    # Separate foreground and unknown class losses
    kd_loss_fg = kd_loss[index_fg].mean() if index_fg.any() else torch.tensor(0.0, requires_grad=True)
    kd_loss_unk = kd_loss[index_unk].mean() if index_unk.any() else torch.tensor(0.0, requires_grad=True)

    # Weighted loss combination
    total_kd_loss = alpha * kd_loss_fg + beta * kd_loss_unk
    return total_kd_loss