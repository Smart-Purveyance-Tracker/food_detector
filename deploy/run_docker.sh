docker run -p $1:9000 \
  --env-file deploy/env.list \
  -v $2:/app/data  \
  food_detector:1.0