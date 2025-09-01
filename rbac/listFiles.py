# import os

# def list_files_recursive(directory):
#     for root, dirs, files in os.walk(directory):
#         print(f'In directory: {root}')
#         for file in files:
#             print(f'  {file}')

# # Example usage
# list_files_recursive('C:/Visit2')
# from datetime import datetime, date

# start_date =input("Enter the start date (YYYY-MM-DD): ")
# end_date = input("Enter the end date (YYYY-MM-DD): ")
# noDays = (date.fromisoformat(end_date) - date.fromisoformat(start_date)).days
# print(noDays)
# # print(datetime.date.weekday(datetime.date.fromisoformat(start_date)))
# # print(datetime.date.weekday(datetime.date.fromisoformat(end_date)))
# # print(datetime.weekday(datetime.date.fromisoformat(end_date)))
# print(date.fromisoformat(start_date).weekday())
# print(date.fromisoformat(end_date).weekday())

# print(date.fromisoformat(start_date).strftime("%A"))
# startDay = date.fromisoformat(start_date).weekday()

# print(date.fromisoformat(end_date).strftime("%A"))
# endDay = date.fromisoformat(end_date).weekday()

# for i in range(noDays):

## --------------------- .csv is generated ----------------------------------
# import pandas as pd
# from datetime import datetime, timedelta
# import calendar

# def get_date_input(prompt):
#     """Get a valid date input from the user"""
#     while True:
#         date_str = input(prompt)
#         try:
#             return datetime.strptime(date_str, '%m/%d/%Y')
#         except ValueError:
#             print("Invalid date format. Please use MM/DD/YYYY format.")

# def get_float_input(prompt):
#     """Get a valid float input from the user"""
#     while True:
#         try:
#             return float(input(prompt))
#         except ValueError:
#             print("Please enter a valid number.")

# def get_int_input(prompt):
#     """Get a valid integer input from the user"""
#     while True:
#         try:
#             return int(input(prompt))
#         except ValueError:
#             print("Please enter a valid integer.")

# def generate_timesheet(start_date, end_date, holidays_count, leaves_count, overtime_hours_dict):
#     """
#     Generate a timesheet between start_date and end_date
    
#     Parameters:
#     start_date (datetime): Start date
#     end_date (datetime): End date
#     holidays_count (int): Number of holidays
#     leaves_count (int): Number of leaves
#     overtime_hours_dict (dict): Dictionary with dates as keys and overtime hours as values
    
#     Returns:
#     pandas.DataFrame: Timesheet dataframe
#     """
    
#     # Initialize lists to store data
#     days = []
#     dates = []
#     projects = []
#     regular_hours = []
#     overtime_list = []
#     total_hours = []
    
#     # Track holidays and leaves to assign
#     holidays_to_assign = holidays_count
#     leaves_to_assign = leaves_count
    
#     # Iterate through each day in the date range
#     current_date = start_date
#     while current_date <= end_date:
#         day_name = calendar.day_name[current_date.weekday()]
#         date_str = current_date.strftime('%m/%d/%Y')
        
#         # Check if it's a weekend
#         if current_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
#             days.append(day_name)
#             dates.append(date_str)
#             projects.append("-")
#             regular_hours.append("-")
#             overtime_list.append("-")
#             total_hours.append("-")
        
#         # Check if it's a holiday
#         elif holidays_to_assign > 0:
#             days.append(day_name)
#             dates.append(date_str)
#             projects.append("Holiday")
#             regular_hours.append("0.00")
#             overtime_list.append("0.00")
#             total_hours.append("0.00")
#             holidays_to_assign -= 1
        
#         # Check if it's a leave day
#         elif leaves_to_assign > 0:
#             days.append(day_name)
#             dates.append(date_str)
#             projects.append("Leave")
#             regular_hours.append("0.00")
#             overtime_list.append("0.00")
#             total_hours.append("0.00")
#             leaves_to_assign -= 1
        
#         # Regular working day
#         else:
#             days.append(day_name)
#             dates.append(date_str)
#             projects.append("PolicyPulse")
            
#             # Check for overtime hours
#             ot_hours = overtime_hours_dict.get(date_str, 0.0)
#             regular_hrs = 8.00
                
#             regular_hours.append(f"{regular_hrs:.2f}")
#             overtime_list.append(f"{ot_hours:.2f}")
#             total_hours.append(f"{regular_hrs + ot_hours:.2f}")
        
#         # Move to next day
#         current_date += timedelta(days=1)
    
#     # Create DataFrame
#     df = pd.DataFrame({
#         'Day': days,
#         'Date': dates,
#         'Project Name': projects,
#         'Regular Hours': regular_hours,
#         'Overtime Hours': overtime_list,
#         'Total': total_hours
#     })
    
#     # Calculate totals
#     regular_total = sum([float(h) for h in regular_hours if h != '-'])
#     overtime_total = sum([float(h) for h in overtime_list if h != '-'])
#     grand_total = regular_total + overtime_total
    
#     # Add totals row
#     totals_row = pd.DataFrame({
#         'Day': [""],
#         'Date': ["Total hours"],
#         'Project Name': [""],
#         'Regular Hours': [f"{regular_total:.2f}"],
#         'Overtime Hours': [f"{overtime_total:.2f}"],
#         'Total': [f"{grand_total:.2f}"]
#     })
    
#     df = pd.concat([df, totals_row], ignore_index=True)
    
#     return df

# def main():
#     print("Timesheet Generator")
#     print("===================")
    
#     # Get user inputs
#     start_date = get_date_input("Enter start date (MM/DD/YYYY): ")
#     end_date = get_date_input("Enter end date (MM/DD/YYYY): ")
    
#     # Validate date range
#     if start_date > end_date:
#         print("Error: Start date must be before end date.")
#         return
    
#     holidays_count = get_int_input("Enter number of holidays: ")
#     leaves_count = get_int_input("Enter number of leaves: ")
    
#     # Get overtime hours
#     overtime_hours_dict = {}
#     print("\nEnter overtime hours (press Enter without date to finish):")
#     while True:
#         date_str = input("Enter date for overtime (MM/DD/YYYY) or press Enter to finish: ")
#         if not date_str:
#             break
        
#         try:
#             date_obj = datetime.strptime(date_str, '%m/%d/%Y')
#             if date_obj < start_date or date_obj > end_date:
#                 print("Date must be within the selected date range.")
#                 continue
                
#             hours = get_float_input(f"Enter overtime hours for {date_str}: ")
#             overtime_hours_dict[date_str] = hours
#         except ValueError:
#             print("Invalid date format. Please use MM/DD/YYYY format.")
    
#     # Generate timesheet
#     timesheet = generate_timesheet(start_date, end_date, holidays_count, leaves_count, overtime_hours_dict)
    
#     # Display the timesheet
#     print("\nGenerated Timesheet:")
#     print("=" * 80)
#     print(timesheet.to_string(index=False))
    
#     # Save to CSV
#     filename = f"timesheet_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
#     timesheet.to_csv(filename, index=False)
#     print(f"\nTimesheet saved to '{filename}'")

# if __name__ == "__main__":
#     main()
    
## --------------------- Excel file is generated ----------------------------------

import pandas as pd
from datetime import datetime, timedelta
import calendar
import os

def get_date_input(prompt):
    """Get a valid date input from the user"""
    while True:
        date_str = input(prompt)
        try:
            return datetime.strptime(date_str, '%m/%d/%Y')
        except ValueError:
            print("Invalid date format. Please use MM/DD/YYYY format.")

def get_float_input(prompt):
    """Get a valid float input from the user"""
    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a valid number.")

def get_int_input(prompt):
    """Get a valid integer input from the user"""
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Please enter a valid integer.")

def get_employee_info():
    """Get employee information from user"""
    print("\nEnter Employee Information:")
    employee_name = input("Employee Name: ")
    manager_name = input("Manager Name: ")
    employee_phone = input("Employee Phone: ")
    employee_email = input("Employee Email: ")
    
    return {
        'name': employee_name,
        'manager': manager_name,
        'phone': employee_phone,
        'email': employee_email
    }

def generate_timesheet_data(start_date, end_date, holidays_count, leaves_count, overtime_hours_dict):
    """
    Generate timesheet data between start_date and end_date
    
    Parameters:
    start_date (datetime): Start date
    end_date (datetime): End date
    holidays_count (int): Number of holidays
    leaves_count (int): Number of leaves
    overtime_hours_dict (dict): Dictionary with dates as keys and overtime hours as values
    
    Returns:
    pandas.DataFrame: Timesheet dataframe
    """
    
    # Initialize lists to store data
    days = []
    dates = []
    projects = []
    regular_hours = []
    overtime_list = []
    total_hours = []
    
    # Track holidays and leaves to assign
    holidays_to_assign = holidays_count
    leaves_to_assign = leaves_count
    
    # Iterate through each day in the date range
    current_date = start_date
    while current_date <= end_date:
        day_name = calendar.day_name[current_date.weekday()]
        date_str = current_date.strftime('%m/%d/%Y')
        
        # Check if it's a weekend
        if current_date.weekday() >= 5:  # 5=Saturday, 6=Sunday
            days.append(day_name)
            dates.append(date_str)
            projects.append("-")
            regular_hours.append("-")
            overtime_list.append("-")
            total_hours.append("-")
        
        # Check if it's a holiday
        elif holidays_to_assign > 0:
            days.append(day_name)
            dates.append(date_str)
            projects.append("Holiday")
            regular_hours.append("0.00")
            overtime_list.append("0.00")
            total_hours.append("0.00")
            holidays_to_assign -= 1
        
        # Check if it's a leave day
        elif leaves_to_assign > 0:
            days.append(day_name)
            dates.append(date_str)
            projects.append("Leave")
            regular_hours.append("0.00")
            overtime_list.append("0.00")
            total_hours.append("0.00")
            leaves_to_assign -= 1
        
        # Regular working day
        else:
            days.append(day_name)
            dates.append(date_str)
            projects.append("PolicyPulse")
            
            # Check for overtime hours
            ot_hours = overtime_hours_dict.get(date_str, 0.0)
            regular_hrs = 8.00
                
            regular_hours.append(f"{regular_hrs:.2f}")
            overtime_list.append(f"{ot_hours:.2f}")
            total_hours.append(f"{regular_hrs + ot_hours:.2f}")
        
        # Move to next day
        current_date += timedelta(days=1)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Date': days,
        'Project Name': dates,
        'Regular Hours': projects,
        'Overtime Hours': regular_hours,
        'Total': overtime_list
    })
    
    # Rename columns to match the image
    df.columns = ['Date', 'Project Name', 'Regular Hours', 'Overtime Hours', 'Total']
    
    # Calculate totals
    regular_total = sum([float(h) for h in regular_hours if h != '-'])
    overtime_total = sum([float(h) for h in overtime_list if h != '-'])
    grand_total = regular_total + overtime_total
    
    # Add totals row
    totals_row = pd.DataFrame({
        'Date': [""],
        'Project Name': ["Total hours"],
        'Regular Hours': [f"{regular_total:.2f}"],
        'Overtime Hours': [f"{overtime_total:.2f}"],
        'Total': [f"{grand_total:.2f}"]
    })
    
    df = pd.concat([df, totals_row], ignore_index=True)
    
    return df

def create_excel_timesheet(start_date, end_date, holidays_count, leaves_count, overtime_hours_dict, employee_info):
    """Create an Excel timesheet with proper formatting"""
    
    # Generate filename
    filename = f"Timesheet {start_date.strftime('%m-%d-%Y')} to {end_date.strftime('%m-%d-%Y')}.xlsx"
    
    # Create a Pandas Excel writer using XlsxWriter as the engine
    writer = pd.ExcelWriter(filename, engine='xlsxwriter')
    workbook = writer.book
    worksheet = workbook.add_worksheet('Timesheet')
    
    # Define formats
    header_format = workbook.add_format({
        'bold': True,
        'font_size': 16,
        'align': 'center',
        'valign': 'vcenter'
    })
    
    subheader_format = workbook.add_format({
        'bold': True,
        'font_size': 14,
        'align': 'center'
    })
    
    company_format = workbook.add_format({
        'bold': True,
        'font_size': 12,
        'align': 'center'
    })
    
    address_format = workbook.add_format({
        'font_size': 10,
        'align': 'center'
    })
    
    table_header_format = workbook.add_format({
        'bold': True,
        'bg_color': '#D3D3D3',
        'border': 1,
        'align': 'center',
        'valign': 'vcenter'
    })
    
    cell_format = workbook.add_format({
        'border': 1,
        'align': 'center'
    })
    
    total_format = workbook.add_format({
        'bold': True,
        'border': 1,
        'align': 'center'
    })
    
    # Write company header
    worksheet.merge_range('A1:F1', 'Tech Pundits Inc', header_format)
    worksheet.merge_range('A2:F2', 'Lake Farrington Plaza II, US-130#310', address_format)
    worksheet.merge_range('A3:F3', 'North Brunswick, NJ-08902', address_format)
    worksheet.merge_range('A4:F4', '', company_format)
    
    # Write timesheet title
    worksheet.merge_range('A5:F5', 'Weekly Time Sheet', subheader_format)
    
    # Generate timesheet data
    timesheet_df = generate_timesheet_data(start_date, end_date, holidays_count, leaves_count, overtime_hours_dict)
    
    # Write table headers
    worksheet.write(6, 0, 'Date', table_header_format)
    worksheet.write(6, 1, 'Project Name', table_header_format)
    worksheet.write(6, 2, 'Regular Hours', table_header_format)
    worksheet.write(6, 3, 'Overtime Hours', table_header_format)
    worksheet.write(6, 4, 'Total', table_header_format)
    
    # Write timesheet data
    for row_num, (_, row_data) in enumerate(timesheet_df.iterrows(), 7):
        for col_num, value in enumerate(row_data):
            if row_num == 7 + len(timesheet_df) - 1:  # Last row (totals)
                worksheet.write(row_num, col_num, value, total_format)
            else:
                worksheet.write(row_num, col_num, value, cell_format)
    
    # Add employee information section
    emp_info_row = len(timesheet_df) + 9
    worksheet.merge_range(emp_info_row, 0, emp_info_row, 2, 'Employee:', table_header_format)
    worksheet.merge_range(emp_info_row, 3, emp_info_row, 5, employee_info['name'], cell_format)
    
    worksheet.merge_range(emp_info_row+1, 0, emp_info_row+1, 2, 'Manager:', table_header_format)
    worksheet.merge_range(emp_info_row+1, 3, emp_info_row+1, 5, employee_info['manager'], cell_format)
    
    worksheet.merge_range(emp_info_row+2, 0, emp_info_row+2, 2, 'Employee phone:', table_header_format)
    worksheet.merge_range(emp_info_row+2, 3, emp_info_row+2, 5, employee_info['phone'], cell_format)
    
    worksheet.merge_range(emp_info_row+3, 0, emp_info_row+3, 2, 'Employee e-mail:', table_header_format)
    worksheet.merge_range(emp_info_row+3, 3, emp_info_row+3, 5, employee_info['email'], cell_format)
    
    # Add signature sections
    sig_row = emp_info_row + 5
    worksheet.merge_range(sig_row, 0, sig_row, 2, 'Employee signature:', table_header_format)
    worksheet.merge_range(sig_row, 3, sig_row, 5, '', cell_format)
    
    worksheet.merge_range(sig_row+1, 0, sig_row+1, 2, 'Date:', table_header_format)
    worksheet.merge_range(sig_row+1, 3, sig_row+1, 5, '', cell_format)
    
    worksheet.merge_range(sig_row+3, 0, sig_row+3, 2, 'Manager signature:', table_header_format)
    worksheet.merge_range(sig_row+3, 3, sig_row+3, 5, '', cell_format)
    
    worksheet.merge_range(sig_row+4, 0, sig_row+4, 2, 'Date:', table_header_format)
    worksheet.merge_range(sig_row+4, 3, sig_row+4, 5, '', cell_format)
    
    # Adjust column widths
    worksheet.set_column('A:A', 12)
    worksheet.set_column('B:B', 15)
    worksheet.set_column('C:C', 15)
    worksheet.set_column('D:D', 15)
    worksheet.set_column('E:E', 10)
    worksheet.set_column('F:F', 10)
    
    # Close the Pandas Excel writer and output the Excel file
    writer.close()
    
    return filename

def main():
    print("Timesheet Generator")
    print("===================")
    
    # Get user inputs
    start_date = get_date_input("Enter start date (MM/DD/YYYY): ")
    end_date = get_date_input("Enter end date (MM/DD/YYYY): ")
    
    # Validate date range
    if start_date > end_date:
        print("Error: Start date must be before end date.")
        return
    
    holidays_count = get_int_input("Enter number of holidays: ")
    leaves_count = get_int_input("Enter number of leaves: ")
    
    # Get employee information
    employee_info = get_employee_info()
    
    # Get overtime hours
    overtime_hours_dict = {}
    print("\nEnter overtime hours (press Enter without date to finish):")
    while True:
        date_str = input("Enter date for overtime (MM/DD/YYYY) or press Enter to finish: ")
        if not date_str:
            break
        
        try:
            date_obj = datetime.strptime(date_str, '%m/%d/%Y')
            if date_obj < start_date or date_obj > end_date:
                print("Date must be within the selected date range.")
                continue
                
            hours = get_float_input(f"Enter overtime hours for {date_str}: ")
            overtime_hours_dict[date_str] = hours
        except ValueError:
            print("Invalid date format. Please use MM/DD/YYYY format.")
    
    # Generate Excel timesheet
    filename = create_excel_timesheet(start_date, end_date, holidays_count, leaves_count, overtime_hours_dict, employee_info)
    
    print(f"\nTimesheet saved to '{filename}'")
    print(f"File location: {os.path.abspath(filename)}")

if __name__ == "__main__":
    main()