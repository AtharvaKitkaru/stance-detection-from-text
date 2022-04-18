from django.db import models
from django.utils.translation import gettext as _

# Create your models here.


class Feedback(models.Model):
    email = models.EmailField(_("email"), max_length=254)
    msg = models.TextField(_("additionalMessage"))
    text = models.TextField(_("text"))
    target = models.TextField(_("target"))
    stance = models.TextField(_("stance"))
