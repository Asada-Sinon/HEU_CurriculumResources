#include <iostream> // 引入输入输出流库
#include <vector> // 引入向量（动态数组）库
#include <string> // 引入字符串处理库
#include <memory> // 引入智能指针库
#include <ctime> // 引入时间库

// 定义Person类，用于存储个人信息
class Person {
public:
    // 构造函数，用于初始化Person对象
    Person(std::string name, std::string studentID, std::string gender, int birthYear, int birthMonth, int birthDay)
        : name_(name), studentID_(studentID), gender_(gender), birthYear_(birthYear), birthMonth_(birthMonth), birthDay_(birthDay) {}

    // 获取姓名的方法
    std::string getName() const { return name_; }
    
    // 打印个人信息的方法
    void printInfo() const {
        std::cout << "姓名: " << name_ << "\n"; // 打印姓名
        std::cout << "学号: " << studentID_ << "\n"; // 打印学号
        std::cout << "性别: " << gender_ << "\n"; // 打印性别
        // 计算并打印年龄
        std::cout << "年龄: " << 2024 - birthYear_ - (birthMonth_ > 10 || (birthMonth_ == 10 && birthDay_ > 11)) << "\n";
        // 其他个人信息的打印...
    }

    // 计算在特定穿着日期的年龄
    int calculateAgeOnWearingDate(const std::string& wearingDate) const {
        int wearingYear = std::stoi(wearingDate.substr(0, 4)); // 获取穿着年份
        int wearingMonth = std::stoi(wearingDate.substr(5, 2)); // 获取穿着月份
        int wearingDay = std::stoi(wearingDate.substr(8, 2)); // 获取穿着日期
        // 返回年龄
        return wearingYear - birthYear_ - (wearingMonth < birthMonth_ || (wearingMonth == birthMonth_ && wearingDay < birthDay_));
    }

private:
    std::string name_; // 姓名
    std::string studentID_; // 学号
    std::string gender_; // 性别
    int birthYear_; // 出生年份
    int birthMonth_; // 出生月份
    int birthDay_; // 出生日期
};

// 定义Clothing类，作为服装的基类
class Clothing {
public:
    // 构造函数，初始化服装的拥有者和穿着日期
    Clothing(std::string owner, std::string wearingDate) : owner_(owner), wearingDate_(wearingDate) {}
    
    virtual ~Clothing() {} // 虚析构函数

    virtual void PrintInfo() const = 0; // 纯虚函数，用于打印服装信息
    virtual float CalculateMaxSize(int age) const = 0; // 纯虚函数，用于计算最大尺码
    void SetMaxSize(float size) { max_size_ = size; } // 设置最大尺码
    std::string GetOwner() const { return owner_; } // 获取拥有者

protected:
    std::string owner_; // 拥有者姓名
    std::string wearingDate_; // 穿着日期
    float max_size_; // 最大尺码
};

// 定义ChildClothing类，继承自Clothing类，表示童装
class ChildClothing : public Clothing {
public:
    // 构造函数，初始化童装的材质及其他信息
    ChildClothing(std::string owner, std::string material, std::string wearingDate)
        : Clothing(owner, wearingDate), material_(material) {}

    // 重写打印信息的方法
    void PrintInfo() const override {
        std::cout << "童装 - 材质: " << material_ << ", 最大尺码: " << max_size_ << ", 穿着年份: " << wearingDate_.substr(0, 4) << std::endl;
    }
    
    // 根据年龄计算最大尺码
    float CalculateMaxSize(int age) const override {
        return 0.3f + 0.02f * (age - 6); // 公式依据年龄计算最大尺码
    }

private:
    std::string material_; // 材质
};

// 定义YouthClothing类，继承自Clothing类，表示青年装
class YouthClothing : public Clothing {
public:
    // 构造函数，初始化青年的装季节及其他信息
    YouthClothing(std::string owner, std::string season, std::string wearingDate)
        : Clothing(owner, wearingDate), season_(season) {}
    
    // 重写打印信息的方法
    void PrintInfo() const override {
        std::cout << "青年装 - 季节: " << season_ << ", 最大尺码: " << max_size_ << ", 穿着年份: " << wearingDate_.substr(0, 4) << std::endl;
    }

    // 根据年龄计算最大尺码
    float CalculateMaxSize(int age) const override {
        return 0.4f + 0.01f * (age - 17); // 公式依据年龄计算最大尺码
    }

private:
    std::string season_; // 季节
};

// 定义MiddleAgedClothing类，继承自Clothing类，表示中年装
class MiddleAgedClothing : public Clothing {
public:
    // 构造函数，初始化中年装的颜色及其他信息
    MiddleAgedClothing(std::string owner, std::string color, std::string wearingDate)
        : Clothing(owner, wearingDate), color_(color) {}
    
    // 重写打印信息的方法
    void PrintInfo() const override {
        std::cout << "中年装 - 颜色: " << color_ << ", 最大尺码: " << max_size_ << ", 穿着年份: " << wearingDate_.substr(0, 4) << std::endl;
    }

    // 根据年龄计算最大尺码
    float CalculateMaxSize(int age) const override {
        return 0.5f + 0.006f * (age - 30); // 公式依据年龄计算最大尺码
    }

private:
    std::string color_; // 颜色
};

// 定义ElderlyClothing类，继承自Clothing类，表示老年装
class ElderlyClothing : public Clothing {
public:
    // 构造函数，初始化老年装的图案及其他信息
    ElderlyClothing(std::string owner, std::string pattern, std::string wearingDate)
        : Clothing(owner, wearingDate), pattern_(pattern) {}

    // 重写打印信息的方法
    void PrintInfo() const override {
        std::cout << "老年装 - 图案: " << pattern_ << ", 最大尺码: " << max_size_ << ", 穿着年份: " << wearingDate_.substr(0, 4) << std::endl;
    }

    // 根据年龄计算最大尺码
    float CalculateMaxSize(int age) const override {
        return 0.6f + 0.004f * (age - 45); // 公式依据年龄计算最大尺码
    }

private:
    std::string pattern_; // 图案
};

// 获取当前年份的方法
int getTodayYear() {
    time_t now = time(0); // 获取当前时间
    tm *ltm = localtime(&now); // 将时间转为本地时间结构
    return 1900 + ltm->tm_year; // 返回当前年份
}

int main() {
    int numClothes; // 存储服装数量
    std::cout << "请输入服装件数: "; // 提示用户输入服装件数
    std::cin >> numClothes; // 读取服装件数
    std::cin.ignore(); // 忽略之前读取的换行符

    // 初始化个人信息
    std::string name = "孔维彬";
    std::string studentID = "2023040620";
    std::string gender = "男";
    int birthYear = 2004;
    int birthMonth = 10;
    int birthDay = 11;
    
    // 创建Person对象
    Person person(name, studentID, gender, birthYear, birthMonth, birthDay);
    person.printInfo(); // 打印个人信息

    std::vector<std::unique_ptr<Clothing>> clothes; // 用于存储服装对象的动态数组
    for (int i = 0; i < numClothes; ++i) {
        std::string wearingDate, type, style; // 定义穿着日期、服装类型和风格
        std::cout << "请输入穿着服装的日期 (格式: YYYY-MM-DD): "; // 提示用户输入穿着日期
        std::getline(std::cin, wearingDate); // 读取穿着日期
        std::cout << "请输入服装类型（童装/青年装/中年装/老年装）: "; // 提示用户输入服装类型
        std::getline(std::cin, type); // 读取服装类型

        // 根据穿着日期计算穿着时的年龄
        int wearingAge = person.calculateAgeOnWearingDate(wearingDate);

        // 创建服装对象的指针
        std::unique_ptr<Clothing> clothing;
        // 根据年龄和类型创建相应的服装对象
        if (type == "童装" && wearingAge <= 12) {
            std::cout << "请输入童装材质（丝绸/棉布）: "; // 提示用户输入童装材质
            std::getline(std::cin, style); // 读取材质
            clothing = std::make_unique<ChildClothing>(person.getName(), style, wearingDate); // 创建童装对象
        } else if (type == "青年装" && wearingAge > 12 && wearingAge <= 35) {
            std::cout << "请输入青年装季节（春季/夏季/秋季/冬季）: "; // 提示用户输入青年装季节
            std::getline(std::cin, style); // 读取季节
            clothing = std::make_unique<YouthClothing>(person.getName(), style, wearingDate); // 创建青年装对象
        } else if (type == "中年装" && wearingAge > 35 && wearingAge <= 60) {
            std::cout << "请输入中年装颜色（红色/蓝色/黑色）: "; // 提示用户输入中年装颜色
            std::getline(std::cin, style); // 读取颜色
            clothing = std::make_unique<MiddleAgedClothing>(person.getName(), style, wearingDate); // 创建中年装对象
        } else if (type == "老年装" && wearingAge > 60) {
            std::cout << "请输入老年装图案（刺绣/印花）: "; // 提示用户输入老年装图案
            std::getline(std::cin, style); // 读取图案
            clothing = std::make_unique<ElderlyClothing>(person.getName(), style, wearingDate); // 创建老年装对象
        } else {
            std::cout << "年龄不符合该服装类型的要求。" << std::endl; // 如果年龄不符合要求，输出提示
            continue; // 跳过当前循环，继续下一次
        }

        // 如果成功创建了服装对象
        if (clothing) {
            float size = clothing->CalculateMaxSize(wearingAge); // 计算最大尺码
            clothing->SetMaxSize(size); // 设置服装的最大尺码
            clothes.push_back(std::move(clothing)); // 将服装对象添加到动态数组中
        }
    }

    // 打印所有服装的信息
    for (const auto& clothing : clothes) {
        clothing->PrintInfo(); // 调用打印信息的方法
    }

    return 0; // 返回0，表示程序结束
}
