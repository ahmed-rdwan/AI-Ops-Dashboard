USE project_management;
GO

-- =====================================================
-- Table: plan
-- (يتم إنشاؤه في البداية لأنه مستخدم في جدول user)
-- =====================================================
CREATE TABLE [plan] (
    id INT IDENTITY(1,1) PRIMARY KEY,
    name VARCHAR(255),
    value DECIMAL(10,2)    -- تعديل: سعر
);

-- =====================================================
-- Table: team
-- (يتم إنشاؤه قبل user)
-- =====================================================
CREATE TABLE team (
    id INT IDENTITY(1,1) PRIMARY KEY,
    name VARCHAR(255),
    description TEXT,
    created_by INT    -- سيتم إضافة FK لاحقاً بعد إنشاء user
);

-- =====================================================
-- Table: user
-- =====================================================
CREATE TABLE [user] (
    id INT IDENTITY(1,1) PRIMARY KEY,
    name VARCHAR(255),
    type VARCHAR(50),
    password VARCHAR(255),
    plan_id INT,
    team_id INT,
    email VARCHAR(255),
    FOREIGN KEY (plan_id) REFERENCES [plan](id),
    FOREIGN KEY (team_id) REFERENCES team(id)
);

-- إضافة FK لعمود created_by في جدول team الآن
ALTER TABLE team
ADD CONSTRAINT FK_team_user_created_by
FOREIGN KEY (created_by) REFERENCES [user](id);

-- =====================================================
-- Table: project
-- =====================================================
CREATE TABLE project (
    id INT IDENTITY(1,1) PRIMARY KEY,
    name VARCHAR(255),
    created_by INT,     -- تعديل: أصبح FK
    description TEXT,
    FOREIGN KEY (created_by) REFERENCES [user](id)
);

-- =====================================================
-- Table: sprint
-- =====================================================
CREATE TABLE sprint (
    id INT IDENTITY(1,1) PRIMARY KEY,
    name VARCHAR(255),
    status VARCHAR(50),
    start_date DATE,
    end_date DATE,
    sprint_goal TEXT,
    project_id INT,
    FOREIGN KEY (project_id) REFERENCES project(id)
);

-- =====================================================
-- Table: backlog
-- =====================================================
CREATE TABLE backlog (
    id INT IDENTITY(1,1) PRIMARY KEY,
    name VARCHAR(255),
    status VARCHAR(50),
    start_date DATE,
    end_date DATE,
    backlog_goal TEXT,
    project_id INT,
    FOREIGN KEY (project_id) REFERENCES project(id)
);

-- =====================================================
-- Table: task
-- =====================================================
CREATE TABLE task (
    id INT IDENTITY(1,1) PRIMARY KEY,
    backlog_id INT,
    assigned BIT DEFAULT 0,
    name VARCHAR(255),
    priority VARCHAR(50),
    description TEXT,
    comment TEXT,
    sprint_id INT,
    status VARCHAR(50),
    FOREIGN KEY (backlog_id) REFERENCES backlog(id),
    FOREIGN KEY (sprint_id) REFERENCES sprint(id)
);

-- =====================================================
-- Table: working_task
-- =====================================================
CREATE TABLE working_task (
    task_id INT,
    team_id INT,
    user_id INT,
    start_date DATE,
    end_date DATE,
    PRIMARY KEY (task_id, team_id, user_id),
    FOREIGN KEY (task_id) REFERENCES task(id),
    FOREIGN KEY (team_id) REFERENCES team(id),
    FOREIGN KEY (user_id) REFERENCES [user](id)
);

-- =====================================================
-- Table: channal
-- =====================================================
CREATE TABLE channal (
    id INT IDENTITY(1,1) PRIMARY KEY,
    name VARCHAR(255),
    created_by INT,
    FOREIGN KEY (created_by) REFERENCES [user](id)
);

-- =====================================================
-- Table: chat
-- =====================================================
CREATE TABLE chat (
    id INT IDENTITY(1,1) PRIMARY KEY,
    name VARCHAR(255),
    channal_id INT,
    user_id INT,
    FOREIGN KEY (channal_id) REFERENCES channal(id),
    FOREIGN KEY (user_id) REFERENCES [user](id)
);

-- =====================================================
-- Table: ticket
-- =====================================================
CREATE TABLE ticket (
    id INT IDENTITY(1,1) PRIMARY KEY,
    name VARCHAR(255),
    description TEXT,
    created_by INT,
    assign_to INT,
    priority VARCHAR(50),
    status VARCHAR(50),
    chat_id INT,        -- تعديل: أصبح FK من chat
    attachment TEXT,
    FOREIGN KEY (created_by) REFERENCES [user](id),
    FOREIGN KEY (assign_to) REFERENCES [user](id),
    FOREIGN KEY (chat_id) REFERENCES chat(id)
);

-- =====================================================
-- Table: stock_item
-- =====================================================
CREATE TABLE stock_item (
    id INT IDENTITY(1,1) PRIMARY KEY,
    name VARCHAR(255),
    category VARCHAR(255),
    quantity INT,
    value DECIMAL(10,2),
    updated_by INT,
    FOREIGN KEY (updated_by) REFERENCES [user](id)
);

-- =====================================================
-- Table: solution
-- =====================================================
CREATE TABLE solution (
    id INT IDENTITY(1,1) PRIMARY KEY,
    ticket_id INT,
    stock_id INT,
    description TEXT,
    FOREIGN KEY (ticket_id) REFERENCES ticket(id),
    FOREIGN KEY (stock_id) REFERENCES stock_item(id)
);







--------------------------------------------------------------Ahmed Radwan-----------------------------------------------------------

USE project_management;
GO

-- =====================================================
-- Table: technician_profile
-- (يخزن ذاكرة الـ AI وإحصائيات الأداء لكل موظف)
-- =====================================================
CREATE TABLE technician_profile (
    user_id INT PRIMARY KEY,
    solved_history_text NVARCHAR(MAX) DEFAULT '', -- تاريخ الكلمات المفتاحية
    keyword_weights NVARCHAR(MAX) DEFAULT '{}',   -- أوزان الخبرة (JSON Text)
    active_tickets INT DEFAULT 0,                 -- عدد التيكتات الحالية
    is_present BIT DEFAULT 1,                     -- الحضور (1=حاضر, 0=غائب)
    total_finished_tickets INT DEFAULT 0,         -- إجمالي المنجز
    total_resolution_time FLOAT DEFAULT 0.0,      -- إجمالي ساعات العمل
    avg_resolution_time FLOAT DEFAULT 0.0,        -- متوسط السرعة
    current_floor INT DEFAULT 1,                  -- مكان الموظف الحالي
    FOREIGN KEY (user_id) REFERENCES [user](id)
);
GO

-- (اختياري) ملء الجدول ببيانات افتراضية للمستخدمين الموجودين حالياً
INSERT INTO technician_profile (user_id)
SELECT id FROM [user]
WHERE id NOT IN (SELECT user_id FROM technician_profile);
GO





USE project_management;
GO

-- التأكد من وجود الجدول، لو مش موجود ننشئه
IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='technician_profile' AND xtype='U')
BEGIN
    CREATE TABLE technician_profile (
        user_id INT PRIMARY KEY,
        solved_history_text NVARCHAR(MAX) DEFAULT '', -- خبرة الموظف (كلمات)
        keyword_weights NVARCHAR(MAX) DEFAULT '{}',   -- أوزان الكلمات (JSON)
        active_tickets INT DEFAULT 0,                 -- عدد التيكتات الحالية
        is_present BIT DEFAULT 1,                     -- 1 = حاضر، 0 = غائب
        total_finished_tickets INT DEFAULT 0,         -- إجمالي المنجز
        total_resolution_time FLOAT DEFAULT 0.0,      -- ساعات العمل
        avg_resolution_time FLOAT DEFAULT 0.0,        -- متوسط السرعة
        current_floor INT DEFAULT 1,                  -- مكانه الحالي
        FOREIGN KEY (user_id) REFERENCES [user](id)
    );
END
GO

------------------------------------------------------------------------------DATA------------------------------------------------------------

USE project_management;
GO

-- 1. تنظيف البيانات القديمة (اختياري - عشان نبدأ على نضافة)
-- DELETE FROM solution;
-- DELETE FROM ticket;
-- DELETE FROM technician_profile;
-- DELETE FROM stock_item;
-- DELETE FROM chat;
-- DELETE FROM channal;
-- DELETE FROM working_task;
-- DELETE FROM task;
-- DELETE FROM backlog;
-- DELETE FROM sprint;
-- DELETE FROM project;
-- DELETE FROM team; -- (يحتاج معالجة خاصة بسبب العلاقة الدائرية)
-- DELETE FROM [user];
-- DELETE FROM [plan];
-- GO

-- =============================================
-- 2. إدخال البيانات الأساسية (Plans & Teams)
-- =============================================
INSERT INTO [plan] (name, value) VALUES 
('Basic', 0.00),
('Premium', 99.99);

-- إدخال تيم مبدئي (بدون مدير حالياً)
INSERT INTO team (name, description) VALUES 
('Infrastructure', 'Network and Servers'),
('Helpdesk', 'User Support'),
('Development', 'Software and ERP');

-- =============================================
-- 3. إدخال الموظفين (Users) - أبطال الـ AI
-- =============================================
DECLARE @TeamInfra INT = (SELECT id FROM team WHERE name = 'Infrastructure');
DECLARE @TeamHelp INT = (SELECT id FROM team WHERE name = 'Helpdesk');
DECLARE @TeamDev INT = (SELECT id FROM team WHERE name = 'Development');
DECLARE @PlanID INT = (SELECT id FROM [plan] WHERE name = 'Basic');

INSERT INTO [user] (name, type, password, plan_id, team_id, email) VALUES 
-- Tech 1: Ahmed (Network Expert)
('Ahmed', 'Technician', '123456', @PlanID, @TeamInfra, 'ahmed@company.com'),
-- Tech 2: Sara (Software Expert)
('Sara', 'Technician', '123456', @PlanID, @TeamDev, 'sara@company.com'),
-- Tech 3: Khaled (Hardware Expert)
('Khaled', 'Technician', '123456', @PlanID, @TeamHelp, 'khaled@company.com'),
-- Tech 4: Mona (General Support)
('Mona', 'Technician', '123456', @PlanID, @TeamHelp, 'mona@company.com'),
-- Manager
('Admin Manager', 'Admin', 'admin123', @PlanID, @TeamInfra, 'admin@company.com');

-- تحديث مدير التيم
UPDATE team SET created_by = (SELECT id FROM [user] WHERE name = 'Admin Manager');

-- =============================================
-- 4. إدخال بيانات الـ AI (Technician Profiles)
-- هام جداً: هنا بنحدد "خبرة" كل واحد عشان الموديل يفهم
-- =============================================
INSERT INTO technician_profile (user_id, solved_history_text, keyword_weights, active_tickets, is_present, current_floor)
VALUES 
(
    (SELECT id FROM [user] WHERE name = 'Ahmed'), 
    'wifi internet router connection lan wan cisco firewall ping signal slow', -- خبرة شبكات
    '{"wifi": 15, "internet": 10, "router": 8, "connection": 5}', -- أوزان عالية
    0, 1, 1 -- الدور الأول، متاح
),
(
    (SELECT id FROM [user] WHERE name = 'Sara'), 
    'password login reset access denied excel word office windows email outlook', -- خبرة سوفت وير
    '{"password": 20, "login": 15, "excel": 10, "windows": 5}', 
    0, 1, 3 -- الدور الثالث
),
(
    (SELECT id FROM [user] WHERE name = 'Khaled'), 
    'printer paper jam toner ink cartridge scanner usb mouse keyboard screen monitor', -- خبرة هاردوير
    '{"printer": 18, "jam": 12, "paper": 10, "mouse": 5}', 
    1, 1, 5 -- الدور الخامس، معاه تيكت واحد
),
(
    (SELECT id FROM [user] WHERE name = 'Mona'), 
    'screen blue cable hdmi display power button', 
    '{"screen": 5, "cable": 3}', 
    3, 1, 2 -- الدور الثاني، مشغولة (3 تيكتات)
);

-- =============================================
-- 5. إدخال البيانات التشغيلية (لعمل النظام)
-- =============================================

-- قنوات ومحادثات (عشان التيكت يحتاج chat_id)
INSERT INTO channal (name, created_by) VALUES ('IT Support Channel', (SELECT id FROM [user] WHERE name = 'Admin Manager'));
DECLARE @ChanID INT = (SELECT TOP 1 id FROM channal);
INSERT INTO chat (name, channal_id, user_id) VALUES ('General Chat', @ChanID, (SELECT id FROM [user] WHERE name = 'Admin Manager'));
DECLARE @ChatID INT = (SELECT TOP 1 id FROM chat);

-- تيكتات مفتوحة (عشان نجرب كشف التكرار Duplicate)
INSERT INTO ticket (name, description, created_by, assign_to, priority, status, chat_id) VALUES 
('Internet Down', 'The wifi is not working in the meeting room', (SELECT id FROM [user] WHERE name = 'Mona'), (SELECT id FROM [user] WHERE name = 'Ahmed'), 'High', 'Open', @ChatID),
('Printer Error', 'Paper jam in HR printer HP LaserJet', (SELECT id FROM [user] WHERE name = 'Sara'), (SELECT id FROM [user] WHERE name = 'Khaled'), 'Medium', 'Open', @ChatID);

-- =============================================
-- 6. إدخال المخزون (Stock Items)
-- هام جداً: عشان الـ Forecasting يشتغل
-- =============================================
INSERT INTO stock_item (name, category, quantity, value, updated_by) VALUES 
('HP LaserJet Toner', 'Ink', 25, 1500.00, (SELECT id FROM [user] WHERE name = 'Admin Manager')), -- كمية قليلة (ممكن يحذر)
('A4 Paper Box', 'Paper', 200, 500.00, (SELECT id FROM [user] WHERE name = 'Admin Manager')), -- كمية كبيرة
('Wireless Mouse', 'Hardware', 10, 200.00, (SELECT id FROM [user] WHERE name = 'Admin Manager')),
('Ethernet Cable 5m', 'Cables', 50, 50.00, (SELECT id FROM [user] WHERE name = 'Admin Manager'));

GO















USE project_management;
GO

-- 1. إنشاء مشروع جديد (Project)
-- نفترض أن الذي أنشأه هو الـ Admin (ID=5 بناءً على البيانات السابقة)
INSERT INTO project (name, created_by, description) 
VALUES ('AI Infrastructure Upgrade', 5, 'Updating company servers and installing AI models');

-- تخزين رقم المشروع في متغير لاستخدامه
DECLARE @ProjectID INT = (SELECT TOP 1 id FROM project ORDER BY id DESC);

-- 2. إنشاء Sprint (Sprint 1) لهذا المشروع
INSERT INTO sprint (name, status, start_date, end_date, sprint_goal, project_id)
VALUES ('Sprint 1', 'Active', GETDATE(), DATEADD(DAY, 14, GETDATE()), 'Setup Basic Infrastructure', @ProjectID);

DECLARE @SprintID INT = (SELECT TOP 1 id FROM sprint ORDER BY id DESC);

-- 3. إنشاء Backlog (اختياري ولكن لربط البيانات)
INSERT INTO backlog (name, status, project_id)
VALUES ('General Backlog', 'Open', @ProjectID);
DECLARE @BacklogID INT = (SELECT TOP 1 id FROM backlog ORDER BY id DESC);

-- 4. إدخال المهام (Tasks) وتوزيعها على الحالات (To Do, In Progress, Completed)
INSERT INTO task (name, priority, status, description, sprint_id, backlog_id, assigned)
VALUES 
-- تاسك في To Do
('Install SQL Server 2022', 'High', 'To Do', 'Install and configure master node', @SprintID, @BacklogID, 0),
-- تاسك في In Progress
('Develop API Endpoints', 'Medium', 'In Progress', 'Create REST API for mobile app', @SprintID, @BacklogID, 1),
-- تاسك في Completed
('Design Database Schema', 'High', 'Completed', 'Finalize tables and relations', @SprintID, @BacklogID, 1);

-- 5. تعيين الموظفين للمهام (Working Task)
-- نحتاج معرفة Task IDs
DECLARE @Task1 INT = (SELECT id FROM task WHERE name = 'Install SQL Server 2022');
DECLARE @Task2 INT = (SELECT id FROM task WHERE name = 'Develop API Endpoints');
DECLARE @Task3 INT = (SELECT id FROM task WHERE name = 'Design Database Schema');

-- تعيين "أحمد" (ID=1) لتاسك السيرفر
INSERT INTO working_task (task_id, team_id, user_id, start_date)
VALUES (@Task1, 1, 1, GETDATE()); -- Team 1 = Infrastructure

-- تعيين "سارة" (ID=2) لتاسك الـ API
INSERT INTO working_task (task_id, team_id, user_id, start_date)
VALUES (@Task2, 3, 2, GETDATE()); -- Team 3 = Development

-- تعيين "المدير" (ID=5) لتاسك الداتابيز المكتملة
INSERT INTO working_task (task_id, team_id, user_id, start_date, end_date)
VALUES (@Task3, 1, 5, GETDATE(), GETDATE());

-- تحديث عمود assigned في جدول task ليصبح 1 (True)
UPDATE task SET assigned = 1 WHERE id IN (@Task1, @Task2, @Task3);

GO




USE project_management;
GO

-- تأكد أولاً أن لديك Sprint رقم 1 (غالباً موجود من الخطوات السابقة)
DECLARE @SprintID INT = (SELECT TOP 1 id FROM sprint);

-- إدخال مهام جديدة "غير موزعة" (Assigned = 0)
INSERT INTO task (name, description, priority, status, assigned, sprint_id, backlog_id)
VALUES 
-- 1. تاسك شبكات (المفروض تروح لأحمد)
('Fix Server Latency', 'The main server connection is very slow and ping is high', 'High', 'To Do', 0, @SprintID, 1),

-- 2. تاسك برمجة (المفروض تروح لسارة)
('Update Python API', 'Refactor the backend code using Pandas and fix login bug', 'Medium', 'To Do', 0, @SprintID, 1),

-- 3. تاسك هاردوير (المفروض تروح لخالد)
('Replace HR Printer Toner', 'HP LaserJet printer needs new ink cartridge and paper jam fix', 'Low', 'To Do', 0, @SprintID, 1);

GO