"""Added alert system columns to User model

Revision ID: 8cd2a968f57f
Revises: 
Create Date: 2024-10-25 01:29:49.252835

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '8cd2a968f57f'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.add_column(sa.Column('alert_sender_email', sa.String(length=120), nullable=True))
        batch_op.add_column(sa.Column('alert_receiver_email', sa.String(length=120), nullable=True))

    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    with op.batch_alter_table('user', schema=None) as batch_op:
        batch_op.drop_column('alert_receiver_email')
        batch_op.drop_column('alert_sender_email')

    # ### end Alembic commands ###