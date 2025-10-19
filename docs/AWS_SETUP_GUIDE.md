# AWS RDS Setup Guide for Indic Annotation Pipeline

This guide walks you through setting up PostgreSQL on AWS RDS for the Indic Annotation Pipeline.

## Prerequisites

- AWS account with appropriate permissions
- AWS CLI installed and configured (optional but recommended)
- PostgreSQL client (`psql`) installed locally

## Step 1: Create AWS RDS PostgreSQL Instance

### Option A: Using AWS Console

1. **Navigate to RDS Console**
   - Go to [AWS RDS Console](https://console.aws.amazon.com/rds/)
   - Click "Create database"

2. **Choose Database Creation Method**
   - Select "Standard create"
   - Engine type: PostgreSQL
   - Version: PostgreSQL 14 or higher (recommended: 15.x)

3. **Templates**
   - For production: "Production"
   - For development/testing: "Dev/Test" or "Free tier" (if eligible)

4. **Settings**
   - DB instance identifier: `indic-annotation-db`
   - Master username: `postgres`
   - Master password: Choose a strong password (save this securely!)

5. **Instance Configuration**
   - For development: `db.t3.micro` or `db.t4g.micro`
   - For production: `db.t3.small` or higher based on workload

6. **Storage**
   - Storage type: General Purpose SSD (gp3)
   - Allocated storage: Start with 20 GB
   - Enable storage autoscaling (optional)
   - Maximum storage threshold: 100 GB

7. **Connectivity**
   - VPC: Default VPC or your custom VPC
   - Public access: **Yes** (for development) or **No** (for production with VPN)
   - VPC security group: Create new or use existing
   - Availability Zone: No preference (or choose specific)

8. **Database Authentication**
   - Password authentication (recommended for simplicity)
   - Or IAM database authentication for enhanced security

9. **Additional Configuration**
   - Initial database name: `indic_annotation`
   - DB parameter group: default
   - Backup retention: 7 days (adjust as needed)
   - Enable encryption: **Yes** (recommended)
   - Enable automated backups: **Yes**

10. **Click "Create database"**
    - Wait 5-10 minutes for instance to be available

### Option B: Using AWS CLI

```bash
aws rds create-db-instance \
    --db-instance-identifier indic-annotation-db \
    --db-instance-class db.t3.micro \
    --engine postgres \
    --engine-version 15.4 \
    --master-username postgres \
    --master-user-password YOUR_STRONG_PASSWORD \
    --allocated-storage 20 \
    --storage-type gp3 \
    --db-name indic_annotation \
    --backup-retention-period 7 \
    --publicly-accessible \
    --storage-encrypted \
    --vpc-security-group-ids sg-xxxxxxxxx \
    --db-subnet-group-name default
```

## Step 2: Configure Security Group

1. **Navigate to EC2 Console → Security Groups**
2. **Find the security group** attached to your RDS instance
3. **Edit Inbound Rules**
   - Type: PostgreSQL
   - Protocol: TCP
   - Port: 5432
   - Source:
     - For development: `0.0.0.0/0` (anywhere) **⚠️ Not recommended for production**
     - For production: Your IP address or VPC CIDR block
4. **Save rules**

## Step 3: Get RDS Endpoint

1. **Go to RDS Console → Databases**
2. **Click on your database instance** (`indic-annotation-db`)
3. **Copy the Endpoint** from "Connectivity & security" tab
   - Example: `indic-annotation-db.c9xyzabc123.us-east-1.rds.amazonaws.com`

## Step 4: Configure Application

1. **Copy `.env.example` to `.env`**:
   ```bash
   cp .env.example .env
   ```

2. **Edit `.env` file** with your RDS details:
   ```env
   DB_HOST=indic-annotation-db.c9xyzabc123.us-east-1.rds.amazonaws.com
   DB_PORT=5432
   DB_NAME=indic_annotation
   DB_USER=postgres
   DB_PASSWORD=your_actual_password
   DB_SSL=true
   DB_POOL_SIZE=5
   DB_MAX_OVERFLOW=10
   ```

## Step 5: Test Connection

1. **Install dependencies**:
   ```bash
   pip install -e .
   ```

2. **Test database connection**:
   ```bash
   python -c "from src.database.connection import test_db_connection; print('✅ Connected!' if test_db_connection() else '❌ Failed')"
   ```

3. **Test with psql** (optional):
   ```bash
   psql -h your-rds-endpoint.us-east-1.rds.amazonaws.com \
        -U postgres \
        -d indic_annotation
   ```

## Step 6: Initialize Database Schema

1. **Option A: Using Python**:
   ```bash
   python -c "from src.database.connection import init_db; init_db()"
   ```

2. **Option B: Using SQL file directly**:
   ```bash
   psql -h your-rds-endpoint.us-east-1.rds.amazonaws.com \
        -U postgres \
        -d indic_annotation \
        -f src/database/schema.sql
   ```

## Step 7: Verify Schema

```bash
psql -h your-rds-endpoint.us-east-1.rds.amazonaws.com \
     -U postgres \
     -d indic_annotation \
     -c "\dt"
```

You should see tables:
- documents
- sentences
- sandhi_annotations
- samasa_annotations
- taddhita_annotations

## Security Best Practices

### 1. **Use Strong Passwords**
   - At least 16 characters
   - Mix of uppercase, lowercase, numbers, and symbols

### 2. **Enable SSL/TLS**
   - Always use `DB_SSL=true` in production
   - Download RDS CA certificate if needed

### 3. **Restrict Network Access**
   - Use VPC with private subnets for production
   - Only allow specific IP addresses in security group
   - Consider using AWS VPN or Direct Connect

### 4. **Enable Encryption**
   - Encryption at rest: Enabled during RDS creation
   - Encryption in transit: SSL/TLS connections

### 5. **Regular Backups**
   - Automated backups: Enabled
   - Manual snapshots before major changes
   - Test restore procedures periodically

### 6. **Monitoring**
   - Enable Enhanced Monitoring
   - Set up CloudWatch alarms for:
     - CPU utilization
     - Database connections
     - Storage space
     - Read/Write IOPS

### 7. **IAM Database Authentication** (Advanced)
   ```python
   # Instead of password, use IAM token
   import boto3

   client = boto3.client('rds')
   token = client.generate_db_auth_token(
       DBHostname=endpoint,
       Port=5432,
       DBUsername='postgres',
       Region='us-east-1'
   )
   ```

## Cost Optimization

1. **Choose Right Instance Size**
   - Start small (db.t3.micro)
   - Scale up based on actual usage

2. **Use Reserved Instances**
   - Save up to 70% with 1-year or 3-year commitment

3. **Stop Development Instances**
   - Stop when not in use (save up to 100%)

4. **Monitor CloudWatch Metrics**
   - Identify underutilized resources

5. **Storage Autoscaling**
   - Only pay for what you use

## Troubleshooting

### Connection Timeout
- Check security group rules
- Verify RDS instance is publicly accessible (if connecting from outside VPC)
- Check if instance is available (not stopped or rebooting)

### Authentication Failed
- Verify username and password
- Check if database name exists
- Ensure password doesn't contain special characters that need escaping

### SSL Connection Error
- Download RDS CA certificate from AWS
- Update connection string with `sslrootcert` parameter

### Performance Issues
- Check CloudWatch metrics
- Increase instance size
- Add read replicas for read-heavy workloads
- Optimize queries and add indexes

## Next Steps

- Set up automated backups
- Configure CloudWatch alarms
- Set up read replicas (for scaling reads)
- Implement connection pooling (already done in `connection.py`)
- Consider Multi-AZ deployment for high availability

## Resources

- [AWS RDS PostgreSQL Documentation](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/CHAP_PostgreSQL.html)
- [RDS Best Practices](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/CHAP_BestPractices.html)
- [PostgreSQL on AWS](https://aws.amazon.com/rds/postgresql/)
