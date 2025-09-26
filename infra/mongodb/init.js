db = db.getSiblingDB('cloudforge');
db.createUser({ user: 'admin', pwd: 'cloudforge2025', roles: [{ role: 'readWrite', db: 'cloudforge' }] });
db.users.insertOne({ email: 'admin@cloudforge.ai', role: 'admin' });

// TEST: MongoDB init script runs
